from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six

from tensor2tensor.utils import registry as t2t_registry
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers, transformer_layers
from tensor2tensor.utils import expert_utils

from tensor2tensor.models.transformer import Transformer, TransformerScorer
from tensor2tensor.models.transformer import transformer_encoder
from tensor2tensor.models.transformer import transformer_prepare_encoder, transformer_prepare_decoder
from tensor2tensor.models.transformer import transformer_ffn_layer, features_to_nonpadding, fast_decode

from tensor2tensor.models.transformer import transformer_base_single_gpu

from . import transformer_with_contexts_layers

import tensorflow as tf

ffn_self_attention_layer = transformer_with_contexts_layers.ffn_self_attention_layer


@t2t_registry.register_model
class TransformerWithContexts(Transformer):

    def body(self, features):
        # concat contexts to inputs
        contexts = {}
        for feature_name in features:
            if 'context' in feature_name and 'raw' not in feature_name:
                contexts[feature_name] = features[feature_name]
        contexts_list = [contexts[feature_name] for feature_name in contexts]
        contexts = tf.concat(contexts_list, axis=1)
        inputs = features["inputs"]
        features["inputs"] = tf.concat([contexts, inputs], axis=1)

        return super(TransformerWithContexts, self).body(features)

    def _fast_decode(self,
                 features,
                 decode_length,
                 beam_size=1,
                 top_beams=1,
                 alpha=1.0):
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.modality["targets"]
        if "targets_segmentation" in features:
            raise NotImplementedError(
                "Decoding not supported on packed datasets "
                " If you want to decode from a dataset, use the non-packed version"
                " of the dataset when decoding.")
        if self.has_input:
            inputs = features["inputs"]
            if target_modality.is_class_modality:
                decode_length = 1
            else:
                decode_length = (
                        common_layers.shape_list(inputs)[1] + features.get(
                    "decode_length", decode_length))

            contexts = {}
            for feature_name in features:
                if 'context' in feature_name and 'raw' not in feature_name:
                    contexts[feature_name] = features[feature_name]

            inputs = tf.expand_dims(inputs, axis=1)
            if len(inputs.shape) < 5:
                inputs = tf.expand_dims(inputs, axis=4)
            s = common_layers.shape_list(inputs)
            batch_size = s[0]
            inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
            # _shard_features called to ensure that the variable names match
            inputs = self._shard_features({"inputs": inputs})["inputs"]
            input_modality = self._problem_hparams.modality["inputs"]

            context_modality = {}
            for context_name in contexts:
                if context_name in self._problem_hparams.modality:
                    context_modality[context_name] = self._problem_hparams.modality[context_name]
                else:
                    context_modality[context_name] = input_modality

            with tf.variable_scope(input_modality.name, reuse=tf.AUTO_REUSE):
                inputs = input_modality.bottom_sharded(inputs, dp)

            for feature_name in contexts:
                with tf.variable_scope(context_modality[feature_name].name, reuse=tf.AUTO_REUSE):
                    contexts[feature_name] = context_modality[feature_name].bottom_sharded(contexts[feature_name], dp)

            contexts_list = [contexts[feature_name][0] for feature_name in contexts]
            contexts = tf.concat(contexts_list, axis=1)
            inputs = [tf.concat([contexts, inputs[0]], axis=1)]

            with tf.variable_scope("body"):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode,
                    inputs,
                    features["target_space_id"],
                    hparams,
                    features=features)
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = None
        else:
            # The problem has no inputs.
            encoder_output = None
            encoder_decoder_attention_bias = None

            # Prepare partial targets.
            # In either features["inputs"] or features["targets"].
            # We force the outputs to begin with these sequences.
            partial_targets = features.get("inputs")
            if partial_targets is None:
                partial_targets = features["targets"]
            assert partial_targets is not None
            partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
            partial_targets = tf.to_int64(partial_targets)
            partial_targets_shape = common_layers.shape_list(partial_targets)
            partial_targets_length = partial_targets_shape[1]
            decode_length = (
                    partial_targets_length + features.get("decode_length", decode_length))
            batch_size = partial_targets_shape[0]

        if hparams.pos == "timing":
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)
        elif hparams.pos == "emb":
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length, hparams.hidden_size]),
                hparams.max_length, "body/targets_positional_embedding", None)
        else:
            positional_encoding = None

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.
            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.
            Args:
              targets: inputs ids to the decoder. [batch_size, 1]
              i: scalar, Step number of the decoding loop.
            Returns:
              Processed targets [batch_size, 1, hidden_dim]
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
            targets = common_layers.flatten4d3d(targets)

            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

            if positional_encoding is not None:
                targets += positional_encoding[:, i:i + 1]
            return targets

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get("encoder_output"),
                    cache.get("encoder_decoder_attention_bias"),
                    bias,
                    hparams,
                    cache,
                    nonpadding=features_to_nonpadding(features, "targets"))

            with tf.variable_scope(target_modality.name):
                logits = target_modality.top_sharded(body_outputs, None, dp)[0]

            ret = tf.squeeze(logits, axis=[1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]

                def forced_logits():
                    return tf.one_hot(
                        tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
                        -1e9)

                ret = tf.cond(
                    tf.less(i, partial_targets_length), forced_logits, lambda: ret)
            return ret, cache

        ret = fast_decode(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=target_modality.top_dimensionality,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length)
        if partial_targets is not None:
            if beam_size <= 1 or top_beams <= 1:
                ret["outputs"] = ret["outputs"][:, partial_targets_length:]
            else:
                ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
        return ret


@t2t_registry.register_model
class DiscourseAwareTransformer(TransformerWithContexts):

    def encode(self, inputs, contexts, target_space, hparams, features=None, losses=None):

        inputs = common_layers.flatten4d3d(inputs)
        _contexts = {}
        for context_name in contexts:
            _contexts[context_name] = common_layers.flatten4d3d(contexts[context_name])

        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            transformer_prepare_encoder(inputs, target_space, hparams, features=features))
        encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)

        context_inputs = {}
        self_ctxt_attention_biases = {}
        encoder_decoder_ctxt_attention_biases = {}


        for context_name in _contexts:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                context_input, self_ctxt_attention_bias, encoder_decoder_ctxt_attention_bias = (
                    transformer_prepare_encoder(_contexts[context_name], target_space, hparams, features=features)
                )
                context_input = tf.nn.dropout(context_input, 1.0 - hparams.layer_prepostprocess_dropout)
                context_inputs[context_name] = context_input
                self_ctxt_attention_biases[context_name] = self_ctxt_attention_bias
                encoder_decoder_ctxt_attention_biases[context_name] = encoder_decoder_ctxt_attention_bias

        encoder_output = discourse_aware_transformer_encoder_with_context(encoder_input, self_attention_bias,
                                                                          context_inputs, self_ctxt_attention_biases,
                                                                          features,
                                                                          hparams,
                                                                          save_weights_to=self.attention_weights,
                                                                          losses=losses)
        return encoder_output, self_attention_bias

    def body(self, features):

        hparams = self._hparams
        losses = []

        contexts = {}
        for feature_name in features:
            if 'context' in feature_name and 'raw' not in feature_name:
                contexts[feature_name] = features[feature_name]
        inputs = features["inputs"]
        target_space = features["target_space_id"]

        encoder_output, encoder_decoder_attention_bias = self.encode(inputs,
                                                                     contexts,
                                                                     target_space,
                                                                     hparams=hparams,
                                                                     features=features,
                                                                     losses=losses)

        targets = features["targets"]
        targets_shape = common_layers.shape_list(targets)
        targets = common_layers.flatten4d3d(targets)

        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(targets, hparams, features=features)

        decoder_output = self.decode(decoder_input,
                                     encoder_output,
                                     encoder_decoder_attention_bias,
                                     decoder_self_attention_bias,
                                     hparams=hparams,
                                     nonpadding=features_to_nonpadding(features, "targets"),
                                     losses=losses)

        expected_attentions = features.get("expected_attentions")
        if expected_attentions is not None:
            attention_loss = common_attention.encoder_decoder_attention_loss(
                expected_attentions, self.attention_weights,
                hparams.expected_attention_loss_type,
                hparams.expected_attention_loss_multiplier)
            return decoder_output, {"attention_loss": attention_loss}

        ret = tf.reshape(decoder_output, targets_shape)
        if losses:
            return ret, {"extra_loss": tf.add_n(losses)}
        else:
            return ret

    def _fast_decode(self,
                     features,
                     decode_length,
                     beam_size=1,
                     top_beams=1,
                     alpha=1.0):
        #dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.modality["targets"]

        inputs = features["inputs"]

        decode_length = (common_layers.shape_list(inputs)[1] + features.get("decode_length", decode_length))

        #inputs = tf.expand_dims(inputs, axis=1)
        #if len(inputs.shape) < 5:
        #    inputs = tf.expand_dims(inputs, axis=4)

        s = common_layers.shape_list(inputs)
        batch_size = s[0]
        #inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
        # _shard_features called to ensure that the variable names match
        #inputs = self._shard_features({"inputs": inputs})["inputs"]
        input_modality = self._problem_hparams.modality["inputs"]
        context_modality = {}

        contexts = {}
        for feature_name in features:
            if 'context' in feature_name and 'raw' not in feature_name:
                contexts[feature_name] = features[feature_name]

        for context_name in contexts:
            if context_name in self._problem_hparams.modality:
                context_modality[context_name] = self._problem_hparams.modality[context_name]
            else:
                context_modality[context_name] = input_modality

        with tf.variable_scope(input_modality.name, reuse=tf.AUTO_REUSE):
            inputs = input_modality.bottom(inputs)
            for context_name in contexts:
                contexts[context_name] = context_modality[context_name].bottom(contexts[context_name])

        with tf.variable_scope("body", reuse=tf.AUTO_REUSE):
            encoder_output, encoder_decoder_attention_bias = self.encode(inputs,
                                                                         contexts,
                                                                         features["target_space_id"],
                                                                         hparams,
                                                                         features=features)
        #encoder_output = encoder_output[0]
        #encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
        partial_targets = None

        if hparams.pos == "timing":
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)
        elif hparams.pos == "emb":
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length + 1, hparams.hidden_size]),
                hparams.max_length, "targets_positional_embedding", None)
        else:
            positional_encoding = None

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.
            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.
            Args:
              targets: inputs ids to the decoder. [batch_size, 1]
              i: scalar, Step number of the decoding loop.
            Returns:
              Processed targets [batch_size, 1, hidden_dim]
            """
            # _shard_features called to ensure that the variable names match
            #targets = self._shard_features({"targets": targets})["targets"]
            with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom(targets)
            targets = common_layers.flatten4d3d(targets)

            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

            if positional_encoding is not None:
                targets += positional_encoding[:, i:i + 1]
            return targets

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            with tf.variable_scope("body"):
                body_outputs = self.decode(targets,
                                           cache.get("encoder_output"),
                                           cache.get("encoder_decoder_attention_bias"),
                                           bias,
                                           hparams,
                                           cache,
                                           nonpadding=features_to_nonpadding(features, "targets"))

            with tf.variable_scope(target_modality.name):
                logits = target_modality.top(body_outputs, None)

            ret = tf.squeeze(logits, axis=[1, 2, 3])
            return ret, cache

        ret = fast_decode(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=target_modality.top_dimensionality,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length)

        return ret


def discourse_aware_transformer_encoder_with_context(encoder_input, encoder_self_attention_bias,
                                                     contexts, context_self_attention_biases,
                                                     features,
                                                     hparams,
                                                     name="discourse_aware_encoder",
                                                     save_weights_to=None,
                                                     make_image_summary=True,
                                                     losses=None):
    input_x = encoder_input
    context_xs = {}
    for context_name in contexts:
        context_xs[context_name] = contexts[context_name]
    context_paddings = {}
    context_nonpaddings = {}
    context_pad_removers = {}

    attention_dropout_broadcast_dims = (
        common_layers.comma_separated_string_to_integer_list(
            getattr(hparams, "attention_dropout_broadcast_dims", "")))

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        input_padding = common_attention.attention_bias_to_padding(encoder_self_attention_bias)
        input_nonpadding = 1.0 - input_padding
        for context_name in context_self_attention_biases:
            context_paddings[context_name] = common_attention.attention_bias_to_padding(
                context_self_attention_biases[context_name])
            context_nonpaddings[context_name] = 1.0 - context_paddings[context_name]

        input_pad_remover = None
        for context_name in context_paddings:
            context_pad_removers[context_name] = None
        if hparams.use_pad_remover and not common_layers.is_xla_compiled():
            input_pad_remover = expert_utils.PadRemover(input_padding)
            for context_name in context_paddings:
                context_pad_removers[context_name] = expert_utils.PadRemover(context_paddings[context_name])

        temp_hparam = tf.contrib.training.HParams() # copy hparams except num_hidden_layers -> num_hidden_layers - 1
        for key, val in hparams.values().items():
            temp_hparam.add_hparam(key, val)
        temp_hparam.set_hparam("num_hidden_layers", hparams.num_hidden_layers - 1)

        encoder_output = transformer_with_contexts_layers.transformer_encoder(
            input_x, encoder_self_attention_bias, temp_hparam,
            nonpadding=features_to_nonpadding(features, "inputs"),
            save_weights_to=save_weights_to, make_image_summary=make_image_summary)


        context_encoded_outputs = {}
        for context_name in context_xs:
            context_encoded_outputs[context_name] = transformer_with_contexts_layers.transformer_encoder(
                context_xs[context_name], context_self_attention_biases[context_name],
                temp_hparam, nonpadding=features_to_nonpadding(features, context_name),
                save_weights_to=save_weights_to, make_image_summary=make_image_summary)

        # last layer
        with tf.variable_scope("encoder/layer_%d" % hparams.num_hidden_layers, reuse=tf.AUTO_REUSE):
            for context_name in context_encoded_outputs:
                with tf.variable_scope(context_name, reuse=tf.AUTO_REUSE):
                    with tf.variable_scope("self_attention", reuse=tf.AUTO_REUSE):
                        _y = common_attention.multihead_attention(
                            common_layers.layer_preprocess(context_encoded_outputs[context_name], hparams),
                            None,
                            context_self_attention_biases[context_name],
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size,
                            hparams.num_heads,
                            hparams.attention_dropout,
                            attention_type=hparams.self_attention_type,
                            save_wieghts_to=save_weights_to,
                            max_relative_position=hparams.max_relative_position,
                            make_image_summary=make_image_summary,
                            max_length=hparams.get("max_length"),
                            vars_3d=hparams.get("attention_variables_3d")
                        )
                        context_encoded_outputs[context_name] = common_layers.layer_postprocess(
                            context_encoded_outputs[context_name], _y, hparams)

                    with tf.variable_scope("ffn", reuse=tf.AUTO_REUSE):
                        _y = transformer_ffn_layer(
                            common_layers.layer_preprocess(context_encoded_outputs[context_name], hparams),
                            hparams,
                            context_pad_removers[context_name],
                            conv_padding="SAME",
                            nonpadding_mask=context_nonpaddings[context_name]
                        )
                        context_encoded_outputs[context_name] = common_layers.layer_postprocess(
                            context_encoded_outputs[context_name], _y, hparams)

                    with tf.variable_scope("context_input_attention"):
                        _y = transformer_with_contexts_layers.multihead_attention(
                            common_layers.layer_preprocess(encoder_output, hparams),
                            context_encoded_outputs[context_name],
                            context_self_attention_biases[context_name],
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size,
                            hparams.num_heads,
                            hparams.attention_dropout,
                            attention_type=hparams.self_attention_type,
                            save_weights_to=save_weights_to,
                            make_image_summary=make_image_summary,
                            max_relative_position=hparams.max_relative_position,
                            dropout_broadcast_dims=attention_dropout_broadcast_dims,
                            max_length=hparams.get("max_length"),
                            vars_3d=hparams.get("attention_variables_3d")
                        )
                    context_encoded_outputs[context_name] = common_layers.layer_postprocess(encoder_output, _y, hparams)

            with tf.variable_scope("input_self_attention"):
                _y = common_attention.multihead_attention(
                    common_layers.layer_preprocess(encoder_output, hparams),
                    None,
                    encoder_self_attention_bias,
                    hparams.attention_key_channels or hparams.hidden_size,
                    hparams.attention_value_channels or hparams.hidden_size,
                    hparams.hidden_size,
                    hparams.num_heads,
                    hparams.attention_dropout,
                    attention_type=hparams.self_attention_type,
                    save_weights_to=save_weights_to,
                    max_relative_position=hparams.max_relative_position,
                    make_image_summary=make_image_summary,
                    dropout_broadcast_dims=attention_dropout_broadcast_dims,
                    max_length=hparams.get("max_length"),
                    vars_3d=hparams.get("attention_variables_3d")
                )
                encoder_output = common_layers.layer_postprocess(encoder_output, _y, hparams)

            with tf.variable_scope("gated_sum"):
                # concat all context_xs first
                context_list = []
                for context_name in context_encoded_outputs:
                    context_list.append(context_encoded_outputs[context_name])

                context_x = tf.add_n(context_list) / len(context_list)

                _depth = common_layers.shape_list(encoder_output)[-1]
                gate = tf.layers.dense(tf.concat([context_x, encoder_output], axis=-1), _depth,
                                       activation=tf.nn.sigmoid)
                if save_weights_to:
                    save_weights_to["gated_sum"] = gate
                encoder_output = gate * encoder_output + (1. - gate) * context_x

            with tf.variable_scope("ffn"):
                _y = transformer_ffn_layer(
                    common_layers.layer_preprocess(encoder_output, hparams),
                    hparams,
                    input_pad_remover,
                    conv_padding="SAME",
                    nonpadding_mask=input_nonpadding,
                    losses=losses
                )
                encoder_output = common_layers.layer_postprocess(encoder_output, _y, hparams)

    return common_layers.layer_preprocess(encoder_output, hparams)

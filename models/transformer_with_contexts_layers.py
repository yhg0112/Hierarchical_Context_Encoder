from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import mlperf_log

from tensor2tensor.layers import transformer_layers

import tensorflow as tf

from tensorflow.python.ops import inplace_ops


def ffn_self_attention_layer(x,
                             filter_depth,
                             output_depth,
                             num_parts,
                             dropout_rate,
                             share_kv=False,
                             name=None,
                             save_weights_to=None):
    """Self-attention feedforward layer.

    We use self-attention to do feedforward computations. We apply this function
    positionwise where for each position, we linearly transform the output to have
    depth filter_depth, and break up the result depth-wise into num_parts
    contiguous parts. The parts self-attend, we concatenate the results
    depth-wise, and we linearly transform to a depth of output_depth. The goal is
    to get multiplicative interactions between components of a representation.

    Args:
      x: a Tensor with shape [batch, length, channels]
      filter_depth: an integer
      output_depth: an integer
      num_parts: an integer dividing filter depth
      dropout_rate: a floating point number
      share_kv: Share the key value transform
      name: an optional string

    Returns:
      A Tensor with shape [batch, length, output_depth].
    """
    with tf.variable_scope(
            name, default_name="feedforward_self_attention", values=[x]):
        x_shape = common_layers.shape_list(x)
        part_depth = filter_depth // num_parts
        if not share_kv:
            combined = common_layers.dense(
                x, filter_depth * 3, use_bias=False, name="qkv_transform")
            combined = tf.expand_dims(combined, axis=2)
            q, k, v = tf.split(combined, 3, axis=3)
        else:
            q = tf.expand_dims(
                common_layers.dense(
                    x, filter_depth, use_bias=False, name="q_transform"),
                axis=2)
            kv_combined = tf.expand_dims(
                common_layers.dense(
                    tf.concat([x, x], axis=1),
                    filter_depth,
                    use_bias=False,
                    name="kv_transform"),
                axis=2)
            k, v = tf.split(kv_combined, [x_shape[1], x_shape[1]], axis=1)

        batch_q = tf.reshape(q, [-1, 1, num_parts, part_depth])
        batch_k = tf.reshape(k, [-1, 1, num_parts, part_depth])
        batch_v = tf.reshape(v, [-1, 1, num_parts, part_depth])

        batch_q *= part_depth**-0.5
        # non-masked bias
        bias = None
        x = dot_product_attention(
            batch_q, batch_k, batch_v, bias, dropout_rate,
            save_weights_to=save_weights_to)
        x = tf.reshape(x, [x_shape[0], x_shape[1], filter_depth])
        x = common_layers.dense(
            x, output_depth, use_bias=False, name="output_transform")
        return x


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True,
                          save_weights_to=None,
                          dropout_broadcast_dims=None):
    """Dot-product attention, multiple attention weights can be saved.

    Args:
      q: Tensor with shape [..., length_q, depth_k].
      k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
        match with q.
      v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
        match with q.
      bias: bias Tensor (see attention_bias())
      dropout_rate: a float.
      image_shapes: optional tuple of integer scalars.
        see comments for attention_image_summary()
      name: an optional string
      make_image_summary: True if you want an image summary.
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      dropout_broadcast_dims: an optional list of integers less than rank of q.
        Specifies in which dimensions to broadcast the dropout decisions.

    Returns:
      Tensor with shape [..., length_q, depth_v].
    """
    with tf.variable_scope(
            name, default_name="dot_product_attention", values=[q, k, v]) as scope:
        logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
        if bias is not None:
            bias = common_layers.cast_like(bias, logits)
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if save_weights_to is not None:
            if scope.name in save_weights_to:
                save_weights_to[scope.name] += [weights]
                save_weights_to[scope.name + "/logits"] += [logits]
            else:
                save_weights_to[scope.name] = [weights]
                save_weights_to[scope.name + "/logits"] = [logits]
        # Drop out attention links for each head.
        weights = common_layers.dropout_with_broadcast_dims(
            weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
        if common_layers.should_generate_summaries() and make_image_summary:
            common_attention.attention_image_summary(weights, image_shapes)
        return tf.matmul(weights, v)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        attn_bias_for_padding=None):
    """A stack of transformer layers, multiple attention weights can be saved.

    Args:
      encoder_input: a Tensor
      encoder_self_attention_bias: bias Tensor for self-attention
         (see common_attention.attention_bias())
      hparams: hyperparameters for model
      name: a string
      nonpadding: optional Tensor with shape [batch_size, encoder_length]
        indicating what positions are not padding.  This must either be
        passed in, which we do for "packed" datasets, or inferred from
        encoder_self_attention_bias.  The knowledge about padding is used
        for pad_remover(efficiency) and to mask out padding in convolutional
        layers.
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      make_image_summary: Whether to make an attention image summary.
      losses: optional list onto which to append extra training losses
      attn_bias_for_padding: Padded attention bias in case a unidirectional
        encoder is being used where future attention is masked.

    Returns:
      y: a Tensors
    """
    x = encoder_input
    attention_dropout_broadcast_dims = (
        common_layers.comma_separated_string_to_integer_list(
            getattr(hparams, "attention_dropout_broadcast_dims", "")))
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
        value=hparams.num_encoder_layers or hparams.num_hidden_layers)
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
        value=hparams.attention_dropout)
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
        value={
            "use_bias": "false",
            "num_heads": hparams.num_heads,
            "hidden_size": hparams.hidden_size
        })

    with tf.variable_scope(name):
        if nonpadding is not None:
            padding = 1.0 - nonpadding
        else:
            attention_bias = encoder_self_attention_bias
            if attn_bias_for_padding is not None:
                attention_bias = attn_bias_for_padding
            padding = common_attention.attention_bias_to_padding(attention_bias)
            nonpadding = 1.0 - padding
        pad_remover = None
        if hparams.use_pad_remover and not common_layers.is_xla_compiled():
            pad_remover = expert_utils.PadRemover(padding)
        for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        common_layers.layer_preprocess(x, hparams),
                        None,
                        encoder_self_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        attention_type=hparams.self_attention_type,
                        max_relative_position=hparams.max_relative_position,
                        heads_share_relative_embedding=(
                            hparams.heads_share_relative_embedding),
                        add_relative_to_values=hparams.add_relative_to_values,
                        save_weights_to=save_weights_to,
                        make_image_summary=make_image_summary,
                        dropout_broadcast_dims=attention_dropout_broadcast_dims,
                        max_length=hparams.get("max_length"),
                        vars_3d=hparams.get("attention_variables_3d"))
                    x = common_layers.layer_postprocess(x, y, hparams)
                with tf.variable_scope("ffn"):
                    y = transformer_layers.transformer_ffn_layer(
                        common_layers.layer_preprocess(x, hparams),
                        hparams,
                        pad_remover,
                        conv_padding="SAME",
                        nonpadding_mask=nonpadding,
                        losses=losses)
                    x = common_layers.layer_postprocess(x, y, hparams)
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        mlperf_log.transformer_print(
            key=mlperf_log.MODEL_HP_NORM,
            value={"hidden_size": hparams.hidden_size})
        return common_layers.layer_preprocess(x, hparams)


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        attention_type="dot_product",
                        max_relative_position=None,
                        heads_share_relative_embedding=False,
                        add_relative_to_values=False,
                        image_shapes=None,
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        vars_3d=False,
                        **kwargs):
    """Multihead scaled-dot-product attention with input/output transformations.

    Args:
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
      bias: bias Tensor (see attention_bias())
      total_key_depth: an integer
      total_value_depth: an integer
      output_depth: an integer
      num_heads: an integer dividing total_key_depth and total_value_depth
      dropout_rate: a floating point number
      attention_type: a string, either "dot_product", "dot_product_relative",
                      "local_mask_right", "local_unmasked", "masked_dilated_1d",
                      "unmasked_dilated_1d", graph, or any attention function
                      with the signature (query, key, value, **kwargs)
      max_relative_position: Maximum distance between inputs to generate
                             unique relation embeddings for. Only relevant
                             when using "dot_product_relative" attention.
      heads_share_relative_embedding: boolean to share relative embeddings
      add_relative_to_values: a boolean for whether to add relative component to
                              values.
      image_shapes: optional tuple of integer scalars.
                    see comments for attention_image_summary()
      block_length: an integer - relevant for "local_mask_right"
      block_width: an integer - relevant for "local_unmasked"
      q_filter_width: An integer specifying how wide you want the query to be.
      kv_filter_width: An integer specifying how wide you want the keys and values
                       to be.
      q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
                 kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
                 no padding.
      cache: dict containing Tensors which are the results of previous
             attentions, used for fast decoding. Expects the dict to contrain two
             keys ('k' and 'v'), for the initial call the values for these keys
             should be empty Tensors of the appropriate shape.
                 'k' [batch_size, 0, key_channels]
                 'v' [batch_size, 0, value_channels]
      gap_size: Integer option for dilated attention to indicate spacing between
                memory blocks.
      num_memory_blocks: Integer option to indicate how many memory blocks to look
                         at.
      name: an optional string.
      save_weights_to: an optional dictionary to capture attention weights
        for vizualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      make_image_summary: Whether to make an attention image summary.
      dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.
      vars_3d: use 3-dimensional variables for input/output transformations
      **kwargs (dict): Parameters for the attention function

    Caching:
      WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
      the caching assumes that the bias contains future masking.

      The caching works by saving all the previous key and value values so that
      you are able to send just the last query location to this attention
      function. I.e. if the cache dict is provided it assumes the query is of the
      shape [batch_size, 1, hidden_dim] rather than the full memory.

    Returns:
      The result of the attention transformation. The output shape is
          [batch_size, length_q, hidden_dim]
      unless the cache dict is provided in which case only the last memory
      position is calculated and the output shape is [batch_size, 1, hidden_dim]
      Optionally returns an additional loss parameters (ex: load balance loss for
      the experts) returned by the attention_type function.

    Raises:
      ValueError: if the key depth or value depth are not divisible by the
        number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_value_depth, num_heads))
    vars_3d_num_heads = num_heads if vars_3d else 0
    with tf.variable_scope(name, default_name="multihead_attention",
                           values=[query_antecedent, memory_antecedent]):

        if cache is None or memory_antecedent is None:
            q, k, v = common_attention.compute_qkv(
                query_antecedent, memory_antecedent,
                total_key_depth, total_value_depth, q_filter_width,
                kv_filter_width, q_padding, kv_padding,
                vars_3d_num_heads=vars_3d_num_heads)
        if cache is not None:
            if attention_type not in ["dot_product", "dot_product_relative"]:
                raise NotImplementedError(
                    "Caching is not guaranteed to work with attention types other than"
                    " dot_product.")
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                                 "for details.")

            if memory_antecedent is not None:
                # Encoder-Decoder Attention Cache
                q = common_attention.compute_attention_component \
                    (query_antecedent, total_key_depth, q_filter_width,
                     q_padding, "q", vars_3d_num_heads=vars_3d_num_heads)
                k = cache["k_encdec"]
                v = cache["v_encdec"]
            else:
                k = common_attention.split_heads(k, num_heads)
                v = common_attention.split_heads(v, num_heads)
                decode_loop_step = kwargs.get("decode_loop_step")
                if decode_loop_step is None:
                    k = cache["k"] = tf.concat([cache["k"], k], axis=2)
                    v = cache["v"] = tf.concat([cache["v"], v], axis=2)
                else:
                    # Inplace update is required for inference on TPU.
                    # Inplace_ops only supports inplace_update on the first dimension.
                    # The performance of current implementation is better than updating
                    # the tensor by adding the result of matmul(one_hot,
                    # update_in_current_step)
                    tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
                    tmp_k = inplace_ops.alias_inplace_update(
                        tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
                    k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
                    tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
                    tmp_v = inplace_ops.alias_inplace_update(
                        tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
                    v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

        q = common_attention.split_heads(q, num_heads)
        if cache is None:
            k = common_attention.split_heads(k, num_heads)
            v = common_attention.split_heads(v, num_heads)

        key_depth_per_head = total_key_depth // num_heads
        if not vars_3d:
            q *= key_depth_per_head**-0.5

        additional_returned_value = None
        if callable(attention_type):  # Generic way to extend multihead_attention
            x = attention_type(q, k, v, **kwargs)
            if isinstance(x, tuple):
                x, additional_returned_value = x  # Unpack
        elif attention_type == "dot_product":
            x = dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
                                      save_weights_to=save_weights_to,
                                      make_image_summary=make_image_summary,
                                      dropout_broadcast_dims=dropout_broadcast_dims)
        else:
            raise NotImplementedError("Only dot product attention is supported.")

        x = common_attention.combine_heads(x)

        # Set last dim specifically.
        x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

        if vars_3d:
            o_var = tf.get_variable(
                "o", [num_heads, total_value_depth // num_heads, output_depth])
            o_var = tf.cast(o_var, x.dtype)
            o_var = tf.reshape(o_var, [total_value_depth, output_depth])
            x = tf.tensordot(x, o_var, axes=1)
        else:
            x = common_layers.dense(
                x, output_depth, use_bias=False, name="output_transform")
        if additional_returned_value is not None:
            return x, additional_returned_value
        return x

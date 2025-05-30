// This file is part of https://github.com/SensorsINI/dnd_hls. 
// This intellectual property is licensed under the terms of the project license available at the root of the project.
#ifndef NNET_EMBED_STREAM_H_
#define NNET_EMBED_STREAM_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "hls_stream.h"

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void embedding(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::embeddings_t embeddings[CONFIG_T::vocab_size * CONFIG_T::n_out])
{
    data_T in_data = data.read();

    InputSequence: for (int j = 0; j < data_T::size; j++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        res_T res_pack;
        #pragma HLS DATA_PACK variable=res_pack

        DenseEmbedding: for (int i = 0; i < CONFIG_T::n_out; i++) {
            #pragma HLS UNROLL
            res_pack[i] = embeddings[in_data[j] * CONFIG_T::n_out + i];
        }
        res.write(res_pack);
    }
}

}

#endif
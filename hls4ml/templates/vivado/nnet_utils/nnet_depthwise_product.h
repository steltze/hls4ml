#ifndef NNET_DEPTHWISE_PRODUCT_H_
#define NNET_DEPTHWISE_PRODUCT_H_

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_product_resource_rf_lt_nchan(data_T data[CONFIG_T::kernel_size * CONFIG_T::n_chan],
                                            res_T res[CONFIG_T::n_chan],
                                            typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan],
                                            typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) {

    const int nin = CONFIG_T::kernel_size * CONFIG_T::n_chan;
    const int nout = CONFIG_T::n_chan;
    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(nin, rufactor);
    const int multiplier_limit = DIV_ROUNDUP(nin, multfactor);
    const int block_factor = DIV_ROUNDUP(nin, rufactor);

    assert((multiplier_limit == block_factor) &&
           "This function is correct only for RF <= N_IN, where N_IN=Kernel_Size*Number_of_Channels");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_RESHAPE   variable=data block factor=block_factor

    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[nout];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int in_index = ir;
        int out_index = ir;

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                    data[in_index], weights[in_index]));

            in_index += rufactor;
            out_index += rufactor;

            if (out_index >= nout) {
                out_index -= nout;
            }
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < nout; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_product_resource_rf_geq_nchan_rem0(
    data_T data[CONFIG_T::kernel_size * CONFIG_T::n_chan], res_T res[CONFIG_T::n_chan],
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) {

    const int nin = CONFIG_T::kernel_size * CONFIG_T::n_chan;
    const int nout = CONFIG_T::n_chan;
    const int rufactor = MIN(CONFIG_T::reuse_factor, nin);
    const int multfactor = MIN(nin, rufactor);
    const int multiplier_limit = DIV_ROUNDUP(nin, multfactor);
    const int block_factor = DIV_ROUNDUP(nin, rufactor);

    assert((rufactor >= nout && rufactor % nout == 0) &&
           "This function is correct only for RF >= N_IN && RF % N_IN == 0, where N_IN=Kernel_Size*Number_of_Channels");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_RESHAPE   variable=data block factor=block_factor

    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[nout];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    int outidx[rufactor];
    int outstep = 0;
IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        outstep++;
        if (outstep == nout) {
            outstep = 0;
        }
    }

    int out_index = 0;

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int in_index = ir;
        out_index = outidx[ir];

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                    data[in_index], weights[in_index]));

            in_index += rufactor;
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < nout; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_product_resource_rf_gt_nchan(data_T data[CONFIG_T::kernel_size * CONFIG_T::n_chan],
                                            res_T res[CONFIG_T::n_chan],
                                            typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan],
                                            typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) {

    const int nin = CONFIG_T::kernel_size * CONFIG_T::n_chan;
    const int nout = CONFIG_T::n_chan;
    const int rufactor = MIN(CONFIG_T::reuse_factor, nin);
    const int block_factor = DIV_ROUNDUP(nin, rufactor);
    assert((rufactor > nout) && "This function is correct only for RF > N_IN, where N_IN=Kernel_Size*Number_of_Channels");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_RESHAPE   variable=data block factor=block_factor

    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[nout];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    const int remainder = CONFIG_T::reuse_factor % nout;

    int outidx[rufactor];
    int outstep = 0;
IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        outstep++;
        if (outstep == nout) {
            outstep = 0;
        }
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int in_index = ir;
        int out_index = outidx[ir];

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            // out_index = in_index % nout;
            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                    data[in_index], weights[in_index]));

            in_index += rufactor;
            out_index += remainder;
            if (out_index >= nout) {
                out_index -= nout;
            }
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < nout; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

} // namespace nnet
#endif

#include <algorithm>
#include <cfloat>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <iterator>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossObjectnessForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[s]);

		// We don't know the target label because we don't have a user click
	  if (has_ignore_label_ && label_value == ignore_label_) {
			const int objectness = static_cast<int>(label[spatial_dim + s]); // Value between 0 and 255
    	const double S0 = max(prob_data[0 * spatial_dim + s], Dtype(FLT_MIN)); // P(class=0) in our model
    	const double P = 1.0 - ((double)objectness / 255.0); // P(class=0) acording to prior
			loss[index] = -P*log(S0) - (1-P)*log(1-S0);
      counts[index] = 1;

		// Supervised; we do know the target label
    } else {
      loss[index] = -log(max(prob_data[label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossObjectnessLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  CHECK_EQ(outer_num_ * inner_num_ * 2, bottom[1]->count(0))
      << "Number of labels must match the number of predictions because there are two channels,"
      << "one for gt labels per pixel and one for objectness labels per pixel; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be 2*N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();

  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossObjectnessForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  if (normalize_) {
    Dtype count;
    caffe_gpu_asum(nthreads, counts, &count);
    loss /= count;
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossObjectnessBackwardGPU(const int nthreads, const Dtype* prob_data, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    //const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[s]);
    
		// We don't know the target label because we don't have a user click
		if (has_ignore_label_ && label_value == ignore_label_) {
			const int objectness = static_cast<int>(label[spatial_dim + s]); // Value between 0 and 255
      const double S0 = max(prob_data[0 * spatial_dim + s], Dtype(FLT_MIN)); // P(bg) in our model
      const double P = 1.0 - ((double)objectness / 255.0); // P(bg) acording to prior
			bottom_diff[0 * spatial_dim + s] -= P; // Case(1): class is bg --> grad[class=0] = S0 - P
			// Iterates over all non-background channels
			for (int c = 1; c < channels; ++c) {
				bottom_diff[c * spatial_dim + s] *= ((P-S0) / (1-S0));
      }
      counts[index] = 1;
    } else { // If we have a user click, the gradient calculation stays the same
      bottom_diff[label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossObjectnessLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();

    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossObjectnessBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(nthreads, counts, &count);
      caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossObjectnessLayer);

}  // namespace caffe

#include <algorithm>
#include <cfloat>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <iterator>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SumWeights(const int nthreads,
          const Dtype* label, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_, const int ignore_location_,
          Dtype* sum_weights) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[s]);
    if (ignore_location_ || (has_ignore_label_ && (label_value == ignore_label_))) { // Unsupervised
			sum_weights[index] = 0;
    } else { // Supervised
			sum_weights[index] = (double)(label[3*spatial_dim + s]) / 255.0;    
    }
  } 
}

template <typename Dtype>
__global__ void SumSupervisedPixels(const int nthreads,
          const Dtype* label, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_, const int ignore_location_, 
          Dtype* num_supervised) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	  const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[s]);
    if (ignore_location_ || (has_ignore_label_ && (label_value == ignore_label_))) { // Unsupervised
      num_supervised[index] = 0;
    } else { // Supervised
			num_supervised[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossExpectationForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_, 
					const bool ignore_objectness_, const bool ignore_location_, const bool ignore_constraint_,
					const bool ignore_rank_, const bool ignore_classes_, 
					const bool normalize_supervised_, const bool normalize_classes_, 
					const bool normalize_constraint_, const bool normalize_objectness_,
					const bool is_siftflow_, 
					int* max_pixels, int num_supervised, double sum_weights) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(label[s]);
		const int channels = dim / spatial_dim;

		// Binary vector length 21 (or 33 if SiftFlow) of which classes (including bg) are in image
    bool class_in_image[33];
		for (int i = 0; i < channels; i++) {
			class_in_image[i] = false;
		}
    int L_plus = 0;
    int c = 0;
    while (true) { // Gets all the unique numbers out of the 3rd channel
      int class_ = static_cast<int>(label[2*spatial_dim + c]);
			if (is_siftflow_) class_ -= 1; // Need to 0-index classes if SiftFlow
			if ((class_ < 0) && is_siftflow_) break; // Doesn't include "0" (or -1) label if SiftFlow -- unlabelled
      class_in_image[class_] = true;
      L_plus++; // Includes background class in count if PASCAL
      if ((class_ == 0) && !is_siftflow_) break; // Includes background class if PASCAL
      c++;
    }
    int L_minus = channels - L_plus;
		if (L_plus == 0) printf("L_plus is 0\n");

		// Gets the number of supervised vs. unsupervised pixels (0 supervised if image-level-labels)
		int num_unsupervised = spatial_dim;
		num_unsupervised -= num_supervised;

		loss[index] = 0;	
		// Add this term no matter what (whether we have supervised pixels or not)
		// Treat max pixel for the class as supervised
		// Loss "classes in image"	
		if (!ignore_classes_) {
			for (int c = 0; c < channels; c++) {
				// Sums over classes in image (or background) where s is max-scoring pixel for the class
				if (class_in_image[c] && (s == max_pixels[c]-1)) { 
					double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
					double classes_term = log(Sc);
					if (normalize_classes_ && (L_plus > 0)) classes_term /= L_plus; // Normalization by number of labels in image
					loss[index] -= classes_term; 
				}
			}
		}
	
		// Loss: supervision
		if (!ignore_location_ && (has_ignore_label_ && (label_value != ignore_label_))) { // Supervised
			int local_label_value = label_value;
			if (is_siftflow_) local_label_value--; // 0-index SiftFlow labels (after the check for 255)
			double Sc = max(prob_data[local_label_value * spatial_dim + s], Dtype(FLT_MIN));
			double weight = 1.0;
			if (!ignore_rank_) weight = (double)(label[3*spatial_dim + s]) / 255.0;	
			double supervised_term = weight * log(Sc);
			if (normalize_supervised_) {
				if (num_supervised == 0) printf("Num supervised is 0\n");
				if (ignore_rank_) supervised_term /= num_supervised;
				else supervised_term /= sum_weights;
			}
			loss[index] -= supervised_term;
		}

		// Loss "constraint" for unsupervised (255) pixels
    if (!ignore_constraint_) {
			for (int c = 0; c < channels; c++) {
				// Sums over classes NOT in image where s is max-scoring pixel for the class
				if (!class_in_image[c] && (s == max_pixels[c]-1)) { 
					double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
					double constraint_term = log(1-Sc);
					if (normalize_constraint_ && (L_minus > 0)) constraint_term /= L_minus; // Normalization by number of labels NOT in image
					loss[index] -= constraint_term; 
				}
			}   
    }

		// Loss "objectness" for unsupervised (255) pixels 
		if (!ignore_objectness_ && (has_ignore_label_ && (label_value == ignore_label_))) {
			const int objectness = static_cast<int>(label[1 * spatial_dim + s]); // Value between 0 and 255
      const double S0 = max(prob_data[0 * spatial_dim + s], Dtype(FLT_MIN)); // P(class=0) in our model
      const double P = 1.0 - ((double)(objectness+1) / 257.0); // P(class=0) acording to prior
			double objectness_term = ((P*log(S0)) + (1-P)*log(1-S0));
			if (normalize_objectness_) objectness_term /= num_unsupervised;
			loss[index] -= objectness_term; 
		}
	}
}

template <typename Dtype>
void SoftmaxWithLossExpectationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;

	if (!ignore_rank_) {
		CHECK_EQ(outer_num_ * inner_num_ * 4, bottom[1]->count(0))
      << "Number of labels must match the number of predictions because there are four channels,"
      << "one for gt labels per pixel, one for objectness labels per pixel, one for unique classes, " 
			<< "and one for rank weights; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be 4*N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
	} else {
		CHECK_EQ(outer_num_ * inner_num_ * 3, bottom[1]->count(0))
      << "Number of labels must match the number of predictions because there are three channels, "
      << "one for gt labels per pixel and one for objectness labels per pixel and one for unique classes; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be 3*N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
	}

	fflush(stdout);
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
	const int channels = dim / inner_num_;

	// ------ Get the max-scoring pixels -----------
	float *prob_data_float;
	int num_elements = inner_num_ * channels;
	cudaMalloc((void**)&prob_data_float, sizeof(float) * num_elements);
	
	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) printf("Cublas create failed\n");

	status = cublasSetVector(num_elements, sizeof(float), prob_data, 1, prob_data_float, 1);
	if (status != CUBLAS_STATUS_SUCCESS) printf("Cublas set vector failed\n");

	int max_pixels[channels]; // Pixel indices for max pixel probability for each class 
	for (int class_ = 0; class_ < channels; class_++) {
    int start_index = class_ * inner_num_;
		int idx_max;
  	status = cublasIsamax(handle, inner_num_, prob_data_float + start_index, 1, &idx_max);
  	if (status != CUBLAS_STATUS_SUCCESS) printf("cublasIsamax failed\n");
		max_pixels[class_] = idx_max;  
  }

	cublasDestroy(handle);
	cudaFree(prob_data_float);
	int *max_pixels1;
	cudaMalloc((void**)&max_pixels1, sizeof(int) * channels);
	status = cublasSetVector(channels, sizeof(int), &max_pixels, 1, max_pixels1, 1);
	// ----------------------------------------------------------------

	// Gets the number of supervised pixels
	SumSupervisedPixels<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, inner_num_, has_ignore_label_, ignore_label_, ignore_location_, counts); 
	Dtype num_supervised_d;
	caffe_gpu_asum(nthreads, counts, &num_supervised_d);	
	int num_supervised = (int) num_supervised_d; 

	// Gets the sum of the weights
	Dtype sum_weights;
	if (!ignore_rank_) {
		SumWeights<<<CAFFE_GET_BLOCKS(nthreads), 
				CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, inner_num_, has_ignore_label_, ignore_label_, ignore_location_, counts);
		Dtype sum_weights;
		caffe_gpu_asum(nthreads, counts, &sum_weights);
	} else {
		sum_weights = 0;
	}

  // NOLINT_NEXT_LINE(whitespace/operators)
	SoftmaxLossExpectationForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, ignore_objectness_, ignore_location_, ignore_constraint_, ignore_rank_, ignore_classes_, normalize_supervised_, normalize_classes_, normalize_constraint_, normalize_objectness_, is_siftflow_, max_pixels1, num_supervised, sum_weights); 
	Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  loss /= outer_num_;
	if (loss == 0) printf("Loss is 0\n");
  
	top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossExpectationBackwardGPU(const int nthreads, const Dtype* prob_data, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, const bool ignore_objectness_, const bool ignore_location_, 
					const bool ignore_constraint_, const bool ignore_rank_, const bool ignore_classes_, 
					const bool normalize_supervised_, const bool normalize_classes_, 
					const bool normalize_constraint_, const bool normalize_objectness_,
					const bool is_siftflow_, 
					int *max_pixels, int num_supervised, double sum_weights) {

  CUDA_KERNEL_LOOP(index, nthreads) {
		const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[s]);
		const int channels = dim / spatial_dim;

		// Binary vector length 21 (or 33 if SiftFlow) of which classes (including bg) are in image
    bool class_in_image[33];
    for (int i = 0; i < channels; i++) {
      class_in_image[i] = false;
    }
    int L_plus = 0;
    int c = 0;
    while (true) { // Gets all the unique numbers out of the 3rd channel
      int class_ = static_cast<int>(label[2*spatial_dim + c]);
      if (is_siftflow_) class_ -= 1; // Need to 0-index classes if SiftFlow
			if ((class_ < 0) && is_siftflow_) break; // Doesn't include "0" (or -1) label if SiftFlow -- unlabelled
      class_in_image[class_] = true;
      L_plus++; // Includes background class in count if PASCAL
      if ((class_ == 0) && !is_siftflow_) break; // Includes background class if PASCAL
      c++;
    }
    int L_minus = channels - L_plus;

		 // Gets the number of supervised vs. unsupervised pixels (0 supervised if image-level-labels)
    int num_unsupervised = spatial_dim;
    num_unsupervised -= num_supervised;

    // Gradient "classes in image"
		// This term was added to loss no matter what
		if (!ignore_classes_) {
			int numMaxFor = 0;
			for (int c = 0; c < channels; c++) {
	    	// Sums numMaxFor over all classes that are present in image and pixel s is maximal
  	    if (class_in_image[c] && (s == max_pixels[c]-1)) {
    	    numMaxFor++;
     	 	}
    	}
    	for (int c = 0; c < channels; c++) {
    		bottom_diff[c*spatial_dim + s] *= numMaxFor;
				if (class_in_image[c] && (s == max_pixels[c]-1)) {
        	bottom_diff[c*spatial_dim + s] -= 1;
      	}
				// Normalize by number of classes in image
				if (normalize_classes_ && (L_plus > 0)) bottom_diff[c*spatial_dim + s] /= L_plus; 
    	}
		} else {
			for (int c = 0; c < channels; c++) {
				bottom_diff[c*spatial_dim + s] = 0;
			}
		}		

		//Gradient: supervision
    if (!ignore_location_ && (has_ignore_label_ && (label_value != ignore_label_))) { // Supervised
			int local_label_value = label_value;
      if (is_siftflow_) local_label_value--; // 0-index SiftFlow labels (after the check for 255)
			double weight = 1.0;
      if (!ignore_rank_) weight = (double)(label[3*spatial_dim + s]) / 255.0;
			for (int c = 0; c < channels; c++) {
				double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
				double supervised_term;
				if (c == local_label_value) { // For supervised pixel, for target class t, gradient is S_t - 1
					supervised_term = weight * (Sc - 1);
				} else { // For supervised pixel, for any other class, gradient is S_t
					supervised_term = weight * Sc;
				}
				if (normalize_supervised_) {
        	if (ignore_rank_) supervised_term /= num_supervised;
        	else supervised_term /= sum_weights;
      	}
        bottom_diff[c*spatial_dim + s] += supervised_term;
			} 
		} // For unsupervised pixel, gradient is 0 for all classes (already set)

		// Gradient "constraint" for unsupervised (255) pixels
    if (!ignore_constraint_) {
			// Calculate R = sum of S_ic / (1-S_ic)
			double R = 0;
			for (int c = 0; c < channels; c++) {
				if (!class_in_image[c] && (s == max_pixels[c]-1)) {
					double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
					R += (Sc / (1-Sc));	
				}
			}

      for (int c = 0; c < channels; c++) {
        double Sc = max(prob_data[c * spatial_dim + s], Dtype(FLT_MIN));
				double constraint_term;
        if (!class_in_image[c] && (s == max_pixels[c]-1)) { 
					constraint_term = -Sc*R + (Sc / (1-Sc));
        } else {
					constraint_term = -Sc*R;
        }
				if (normalize_constraint_) constraint_term /= L_minus;
        bottom_diff[c*spatial_dim + s] += constraint_term;
      }
    }

		// Gradient "objectness" for unsupervised (255) pixels
		if (!ignore_objectness_ && (has_ignore_label_ && (label_value == ignore_label_))) {
			const int objectness = static_cast<int>(label[1 * spatial_dim + s]); // Value between 0 and 255
  	  const double S0 = max(prob_data[0 * spatial_dim + s], Dtype(FLT_MIN)); // P(class=0) in our model
	    const double P = 1.0 - ((double)(objectness+1) / 257.0); // P(class=0) acording to prior
		
			for (int c = 0; c < channels; c++) {
				double objectness_term;
				if (c == 0) { // Background
					objectness_term = S0 - P;
				} else { 
					double Sc = max(prob_data[c*spatial_dim + s], Dtype(FLT_MIN));
					objectness_term = Sc*((P-S0)/(1-S0));		
				}
				if (normalize_objectness_) objectness_term /= num_unsupervised;
				bottom_diff[c*spatial_dim + s] += objectness_term;
			}
			// For supervised pixels, gradient objectness is 0 (already done)
		} 

	}
}

template <typename Dtype>
void SoftmaxWithLossExpectationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
		const int channels = dim / inner_num_;

		// ------ Get the max-scoring pixels -----------
  	float *prob_data_float;
  	int num_elements = inner_num_ * channels;
  	cudaMalloc((void**)&prob_data_float, sizeof(float) * num_elements);

  	cublasStatus_t status;
  	cublasHandle_t handle;
  	status = cublasCreate(&handle);
  	if (status != CUBLAS_STATUS_SUCCESS) printf("Cublas create failed\n");

  	status = cublasSetVector(num_elements, sizeof(float), prob_data, 1, prob_data_float, 1);
  	if (status != CUBLAS_STATUS_SUCCESS) printf("Cublas set vector failed\n");

  	int max_pixels[channels]; // Pixel indices for max pixel probability for each class 
  	for (int class_ = 0; class_ < channels; class_++) {
    	int start_index = class_ * inner_num_;
    	int idx_max;
    	status = cublasIsamax(handle, inner_num_, prob_data_float + start_index, 1, &idx_max);
    	if (status != CUBLAS_STATUS_SUCCESS) printf("cublasIsamax failed\n");
    	max_pixels[class_] = idx_max;
  	}

  	cublasDestroy(handle);
  	cudaFree(prob_data_float);
  	int *max_pixels1;
  	cudaMalloc((void**)&max_pixels1, sizeof(int) * channels);
	  status = cublasSetVector(channels, sizeof(int), &max_pixels, 1, max_pixels1, 1);
		// --------------------------------------------------------

		// Gets the number of supervised pixels
	  SumSupervisedPixels<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
  	    CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, inner_num_, has_ignore_label_, ignore_label_, ignore_location_, counts);
  	Dtype num_supervised_d;
  	caffe_gpu_asum(nthreads, counts, &num_supervised_d);
		int num_supervised = (int) num_supervised_d;

		// Gets the sum of the weights
		Dtype sum_weights;
  	if (!ignore_rank_) {
			SumWeights<<<CAFFE_GET_BLOCKS(nthreads),
    	  	CAFFE_CUDA_NUM_THREADS>>>(nthreads, label, inner_num_, has_ignore_label_, ignore_label_, ignore_location_, counts);
  		caffe_gpu_asum(nthreads, counts, &sum_weights);
		} else {
			sum_weights = 0;
		}

    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossExpectationBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, ignore_objectness_, ignore_location_, ignore_constraint_, ignore_rank_, ignore_classes_, normalize_supervised_, normalize_classes_, normalize_constraint_, normalize_objectness_, is_siftflow_, max_pixels1, num_supervised, sum_weights);
  
	  const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossExpectationLayer);

}  // namespace caffe

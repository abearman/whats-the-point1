#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossExpectationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
	ignore_objectness_ = this->layer_param_.loss_param().ignore_objectness();
	ignore_location_ = this->layer_param_.loss_param().ignore_location();
	ignore_constraint_ = this->layer_param_.loss_param().ignore_constraint();
	ignore_classes_ = this->layer_param_.loss_param().ignore_classes();
	ignore_rank_ = this->layer_param_.loss_param().ignore_rank();
	normalize_supervised_ = this->layer_param_.loss_param().normalize_supervised();
	normalize_classes_ = this->layer_param_.loss_param().normalize_classes();
	normalize_constraint_ = this->layer_param_.loss_param().normalize_constraint();
	normalize_objectness_ = this->layer_param_.loss_param().normalize_objectness();
	is_siftflow_ = this->layer_param_.loss_param().is_siftflow();
}

template <typename Dtype>
void SoftmaxWithLossExpectationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
	LOG(INFO) << "outer num: " << outer_num_ << ", inner num: " << inner_num_ << ", bottom[0]->count(): " << bottom[0]->count() << ", bottom[1]->count(): " << bottom[1]->count(0); 
	//int mult = outer_num_ * inner_num_ * 2;
  //CHECK_EQ(mult, bottom[1]->count(0))
    //  << "Number of labels must match the number of predictions because there are two channels,"
	//		<< "one for gt labels per pixel and one for objectness labels per pixel; "
   //   << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
    //  << "label count (number of labels) must be 2*N*H*W, "
     // << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossExpectationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
	printf("Got to cpu forward pass\n");
	const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  //int dim = prob_.count() / outer_num_;
  LOG(INFO) << "inner num: " << inner_num_ << ", bottom[1]->count(): " << bottom[1]->count();

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

	int count = 0;
  Dtype loss = 0;
	
	vector<int> present_classes; 
	int len = 0;
	while (true) { // Gets all the unique numbers out of the 3rd channel
		int class_ = static_cast<int>(label[2*inner_num_ + len]);
		if (class_ != 0) {
			present_classes.push_back(class_);
		} else {
 			break;
		}
		len++;
	}

	// Iterates over all pixels in the image
	for (int j = 0; j < inner_num_; j++) {
		const int label_value = static_cast<int>(label[j]);
		
		// Either:
		// (a) Ignore location (i.e. all pixels are unsupervised), or
		// (b) We don't know the target label because we don't have a user click
		if (has_ignore_label_ && label_value == ignore_label_) { 
				const int objectness = static_cast<int>(label[inner_num_ + j]); // A value beween 0 and 255. Objectness
				double S0 = std::max(prob_data[0 * inner_num_ + j], Dtype(FLT_MIN)); // P(background) in our model
        double P = 1.0 - ((double)objectness/255.0); // P(background) according to objectness prior
				double a0 = P;
				double a_c = (1-P) / present_classes.size();
				double sum_ = 0;
			
				// Sum background into sum_
				sum_ += a0 * S0;

				for (typename vector<int>::iterator iter = present_classes.begin(); iter != present_classes.end(); iter++) {
					int class_ = *iter; // *iter is the class
					double Sc = std::max(prob_data[class_ * inner_num_ + j], Dtype(FLT_MIN)); // P(class=c)  
					sum_ += a_c * Sc;
				}
				loss -= log( sum_ );			
		
		// Supervised, we do know the target label
    } else {
			DCHECK_GE(label_value, 0);
 	  	DCHECK_LT(label_value, prob_.shape(softmax_axis_));
			loss -= log(std::max(prob_data[label_value * inner_num_ + j], Dtype(FLT_MIN)));
		}
		++count;
	
}
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossExpectationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    //int dim = prob_.count() / outer_num_;
    int count = 0;
  
		vector<int> present_classes;
	  int len = 0;
	  while (true) { // Gets all the unique numbers out of the 3rd channel
    	int class_ = static_cast<int>(label[2*inner_num_ + len]);
    	if (class_ != 0) {
    	  present_classes.push_back(class_);
    	} else {
    	  break;
    	}
			len++;
  	}

		for (int j = 0; j < inner_num_; j++) {
			const int label_value = static_cast<int>(label[j]);
			const int objectness = static_cast<int>(label[inner_num_ + j]);
		
			// Either:
    	// (a) Ignore location (i.e. all pixels are unsupervised), or
    	// (b) We don't know the target label because we don't have a user click	
			if (has_ignore_label_ && label_value == ignore_label_) { 
				double S0 = std::max(prob_data[0 * inner_num_ + j], Dtype(FLT_MIN)); // P(background) in our model
        double P = 1.0 - ((double)objectness/255.0); // P(background) according to objectness prior
				
				// Iterates over all channels (including background)
				for (int i = 0; i < bottom[0]->shape(softmax_axis_); ++i) { 
					// Case (1): class c is in the image (or background) 
					if (/*i == 0 ||*/ find(present_classes.begin(), present_classes.end(), i) != present_classes.end()) {
	          double a0 = P;
		        double a_c = (1-P) / present_classes.size();
    		    double sum_ = 0;

      		  // Sum background into sum_
        		sum_ += a0 * S0;

        		for (typename vector<int>::iterator iter = present_classes.begin(); iter != present_classes.end(); iter++) {
          		int class_ = *iter; // *iter is the class
          		double Sc = std::max(prob_data[class_ * inner_num_ + j], Dtype(FLT_MIN)); // P(class=i)  
		          sum_ += a_c * Sc;
        		}
						
						double a_i;
						if (i == 0) a_i = a0; // background
						else a_i = a_c; // not background			
						bottom_diff[i * inner_num_ + j] *= (1 - (a_i / sum_));
					} 
					// Case (2): class c is not in the image, --> grad[class=c] = S_c
					// Do nothing; we already copied it over
				}
			} else { // If we have a user click, the gradient calculation stays the same
				bottom_diff[label_value * inner_num_ + j] -= 1; // This only happens for the target label
			}
			++ count;
		} 

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
		if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossExpectationLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossExpectationLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLossExpectation);

}  // namespace caffe

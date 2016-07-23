#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossObjectnessPresenceLayer<Dtype>::LayerSetUp(
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
}

template <typename Dtype>
void SoftmaxWithLossObjectnessPresenceLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
	//LOG(INFO) << "outer num: " << outer_num_ << ", inner num: " << inner_num_ << ", bottom[0]->count(): " << bottom[0]->count() << ", bottom[1]->count(): " << bottom[1]->count(0); 
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
void SoftmaxWithLossObjectnessPresenceLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  //int dim = prob_.count() / outer_num_;
  LOG(INFO) << "inner num: " << inner_num_ << ", bottom[1]->count(): " << bottom[1]->count();
	int mult = outer_num_ * inner_num_ * 2;
  CHECK_EQ(mult, bottom[1]->count(0))
      << "Number of labels must match the number of predictions because there are two channels,"
      << "one for gt labels per pixel and one for objectness labels per pixel; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be 2*N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
	int count = 0;
  Dtype loss = 0;
	
	vector<Dtype> present_classes (label, label + inner_num_); // Gets the first channel of gt 
	// Gets all unique elements
	sort( present_classes.begin(), present_classes.end() );
	present_classes.erase( unique( present_classes.begin(), present_classes.end() ), present_classes.end() );
	typename vector<Dtype>::iterator it;
	present_classes.erase(remove(present_classes.begin(), present_classes.end(), 255), present_classes.end()); // Removes 255
	LOG(INFO) << "size of present classes: " << present_classes.size() << ", present_classes contains:";
	for (it=present_classes.begin(); it!=present_classes.end(); ++it) {
		LOG(INFO) << ' ' << *it;
	}

	for (int j = 0; j < inner_num_; j++) {
		const int label_value = static_cast<int>(label[j]);
		// We don't know the target label because we don't have a user click
		if (has_ignore_label_ && label_value == ignore_label_) { 
				// Modify this block
				const int objectness = static_cast<int>(label[inner_num_ + j]); // A value beween 0 and 255. Objectness
				double S0 = std::max(prob_data[0 * inner_num_ + j], Dtype(FLT_MIN)); // P(background) in our model
        double P = 1.0 - ((double)objectness/255.0); // P(background) according to objectness prior
				double Q = 0;
				for (typename vector<Dtype>::iterator iter = present_classes.begin(); iter != present_classes.end(); iter++) {
					int c = *iter; // *iter is the class
					double Sc = std::max(prob_data[c * inner_num_ + j], Dtype(FLT_MIN)); // P(class=c)  
					Q += Sc;
				}
				loss -= (P*log(S0) + (1-P)*log(Q)); 
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
void SoftmaxWithLossObjectnessPresenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  	
		// Gets all unique elements
	  vector<Dtype> present_classes (label, label + inner_num_); // Gets the first channel of gt
		sort( present_classes.begin(), present_classes.end() );
  	present_classes.erase( unique( present_classes.begin(), present_classes.end() ), present_classes.end() );
  	typename vector<Dtype>::iterator it;
 		present_classes.erase(remove(present_classes.begin(), present_classes.end(), 255), present_classes.end()); // Removes 255
 		LOG(INFO) << "size of present classes: " << present_classes.size() << ", present_classes contains:";
  	for (it=present_classes.begin(); it!=present_classes.end(); ++it) {
    	LOG(INFO) << ' ' << *it;
  	}

		for (int j = 0; j < inner_num_; j++) {
			const int label_value = static_cast<int>(label[j]);
			const int objectness = static_cast<int>(label[inner_num_ + j]);
			if (has_ignore_label_ && label_value == ignore_label_) { // If we don't have a user click
				//double S0 = std::max(prob_data[0 * inner_num_ + j], Dtype(FLT_MIN)); // P(background) in our model
        double P = 1.0 - ((double)objectness/255.0); // P(background) according to objectness prior
				bottom_diff[0 * inner_num_ + j] -= P; // Case (1): class is bg --> grad[class=0] = S0 - P
				// Iterates over all non-background channels
				for (int c = 1; c < bottom[0]->shape(softmax_axis_); ++c) { 
					 // Case (2): class c is in the image, not bg --> grad[class=c] = ((P+Q-1)/Q) * S_c
					if (find(present_classes.begin(), present_classes.end(), c) != present_classes.end()) {
						double Q = 0;
        		for (typename vector<Dtype>::iterator iter = present_classes.begin(); iter != present_classes.end(); iter++) {
          		int class_ = *iter; // *iter is the class
          		double Sc = std::max(prob_data[class_ * inner_num_ + j], Dtype(FLT_MIN)); // P(class=c)  
          		Q += Sc;
        		}
						bottom_diff[c * inner_num_ + j] *= ((P+Q-1) / Q);
					} 
					// Case (3): class c is not in the image, not bg --> grad[class=c] = S_c
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
STUB_GPU(SoftmaxWithLossObjectnessPresenceLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossObjectnessPresenceLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLossObjectnessPresence);

}  // namespace caffe

#include "simcc_decoder.h"

#include <algorithm>

namespace visionpipe {

void SimccDecoder::decode(const float* simcc_x, const float* simcc_y,
                          int num_keypoints, int x_bins, int y_bins,
                          float split_ratio,
                          std::vector<Keypoint>& keypoints) {
    keypoints.resize(num_keypoints);

    for (int k = 0; k < num_keypoints; ++k) {
        const float* row_x = simcc_x + static_cast<size_t>(k) * x_bins;
        const float* row_y = simcc_y + static_cast<size_t>(k) * y_bins;

        const float* max_x_it = std::max_element(row_x, row_x + x_bins);
        const float* max_y_it = std::max_element(row_y, row_y + y_bins);

        const float max_x = *max_x_it;
        const float max_y = *max_y_it;

        Keypoint& kp = keypoints[k];
        kp.x = static_cast<float>(max_x_it - row_x) / split_ratio;
        kp.y = static_cast<float>(max_y_it - row_y) / split_ratio;
        kp.score = 0.5f * (max_x + max_y);
        if (kp.score < 0.0f) {
            kp.score = 0.0f;
        }
    }
}

}  // namespace visionpipe

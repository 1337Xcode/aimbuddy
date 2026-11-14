#ifndef AIMBUDDY_DETECTOR_DETECTOR_H
#define AIMBUDDY_DETECTOR_DETECTOR_H

#include "../capture/roi_capture.h"
#include "../core/detection_batch.h"
#include "../core/model_profile.h"

namespace aimbuddy {

/// The shared contract of the NCNN and TFLite model runners. One pipeline thread calls a detector.
class Detector {
public:
    virtual ~Detector() = default;
    virtual bool isLoaded() const = 0;
    virtual void unload() = 0;
    virtual bool detect(const RoiFrame& frame, DetectionBatch& batch) = 0;
    virtual void setMinScore(float score) = 0;
    virtual int inputSize() const = 0;
    virtual const ModelProfile& profile() const = 0;
    virtual const char* backendName() const = 0;
};

}  // namespace aimbuddy

#endif  // AIMBUDDY_DETECTOR_DETECTOR_H

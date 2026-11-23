# Training pipeline acceptance scenarios

Use the real local images, pretrained backbone, training loop and ONNX Runtime.
Run in a separate artifact directory; preserve the original splits and checkpoints.
Every run saves commands, hashes, per-image results, logs and a final report.

Before implementation, the required failure cases are:

1. Rebuilding after new images must preserve established validation/test assignments.
   A source/template/near-duplicate group connecting two established splits must
   fail with a reviewable conflict report, without rewriting the split CSVs.
   Exact duplicate images with contradictory labels must not silently favor positives.
2. The calibration set must be separate from train, model-selection val and test.
   A legacy dataset may reserve calibration groups from train once; val/test stay fixed.
3. A real two-stage training run must keep frozen backbone parameters AND BatchNorm
   buffers unchanged in stage 1, then update only the intended unfrozen parameters.
   Whole-image labels must not be retained after randomly cropping/erasing the target.
4. Validation must use FP32 regardless of training AMP. Selection must enforce the
   configured recall/precision/FPR limits, including negative subcategories.
   Infeasible operating points must fail explicitly, never silently choose F-beta.
5. Teacher and student must see the same augmented image when distillation is enabled;
   contradictory teacher predictions must not overrule reviewed hard labels.
6. Calibration must record checkpoint/data provenance and reject overlapping splits.
   Export must check the full calibration set, report per-image numerical/decision
   differences, and calibrate the threshold on the final ONNX file. A failed export
   must leave a previously accepted model untouched.
7. Actual prediction/evaluation CLIs must reject mismatched model/calibration pairs.
   X.zip remains a fixed diagnostic regression set; no claim of improved accuracy
   is made from a short pipeline smoke run.
8. Moving the dataset and code to a different checkout must preserve every image,
   label and split assignment. Rebuilding must keep identical CSVs and relative
   split-state identities. Removing source-machine metadata must not change model
   weights or inference results, and calibration hashes must bind the published files.

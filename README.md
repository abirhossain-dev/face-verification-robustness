# Face verification robustness

How robust are pretrained face-verification models to image degradations? Work in progress: results will be added after the Colab run.

Setup: facenet-pytorch InceptionResnetV1 (VGGFace2 and CASIA-WebFace), LFW 10-fold pairs, same degradation on both images, threshold chosen on 9 folds. Degradations: Gaussian blur, Gaussian noise, JPEG, downsampling, lower-face occlusion, 5 severities each.

Run: python robustness_eval.py

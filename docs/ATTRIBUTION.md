# Attribution

This project uses pretrained models and public sample images. No model training
was performed.

## Models

| Component | Source | License |
|---|---|---|
| Real-ESRGAN x4plus | [Xintao Wang et al., Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause |
| GFPGAN v1.4 (optional) | [Tencent ARC, GFPGAN](https://github.com/TencentARC/GFPGAN) | Apache-2.0 |
| BLIP base | [Salesforce, BLIP](https://github.com/salesforce/BLIP) | BSD-3-Clause |
| SIGGRAPH 2017 colorizer | [Richard Zhang et al., Colorful Image Colorization](https://github.com/richzhang/colorization) | BSD-2-Clause |
| OpenCV YuNet face detector | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) | Apache-2.0 |

The colorizer architecture in `src/colorizers/` is vendored from
`richzhang/colorization`. Its original copyright notice is retained by the
upstream license; see the repository for details.

## Sample images

- `assets/sample_portrait.jpg`: OpenCV's classic `lena.jpg` test image
  ([OpenCV samples](https://github.com/opencv/opencv/tree/4.x/samples/data)),
  used only as a local test image.
- `assets/sample_scene.jpg`: downloaded from [Lorem Picsum](https://picsum.photos/)
  for demo purposes.
- Evaluation crops: scikit-image's bundled public sample images
  (`astronaut`, `chelsea`, `coffee`, `rocket`, `horse`, `camera`).

Generated synthetic old-photo variants and restored outputs are derivative
works created for this project.

## Libraries

PyTorch, torchvision, Transformers, Streamlit, OpenCV, scikit-image, LPIPS,
spandrel, Pillow, NumPy, and their dependencies retain their respective licenses.

## Citation

If you use this project in a report, cite the upstream models:

```bibtex
@inproceedings{wang2021realesrgan,
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={International Conference on Computer Vision Workshops (ICCVW)},
  year={2021}
}

@inproceedings{wang2021gfpgan,
  title={Towards Real-World Blind Face Restoration with Generative Facial Prior},
  author={Wang, Xintao and Li, Yu and Zhang, Honglun and Shan, Ying},
  booktitle={CVPR},
  year={2021}
}

@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  booktitle={ICML},
  year={2022}
}

@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A.},
  booktitle={ECCV},
  year={2016}
}
```

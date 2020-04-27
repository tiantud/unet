* origional unet, 100 epochs

lr = 0.001 | lr = 0.0001 | lr = 0.00001
------------ | ------------- | -------------
![val_n1](20200416_223905_normal_unet_Adam_e100_lr0.001/val_n1.svg) | ![val_n1](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n1.svg) |![val_n1](20200420_184537_normal_unet_e100_lr1e-05/val_n1.svg)
![val_n2](20200416_223905_normal_unet_Adam_e100_lr0.001/val_n2.svg) | ![val_n2](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n2.svg) |![val_n2](20200420_184537_normal_unet_e100_lr1e-05/val_n2.svg)
![val_n3](20200416_223905_normal_unet_Adam_e100_lr0.001/val_n3.svg) | ![val_n3](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n3.svg) |![val_n3](20200420_184537_normal_unet_e100_lr1e-05/val_n3.svg)
![val_n4](20200416_223905_normal_unet_Adam_e100_lr0.001/val_n4.svg) | ![val_n4](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n4.svg) |![val_n4](20200420_184537_normal_unet_e100_lr1e-05/val_n4.svg)
![val_n5](20200416_223905_normal_unet_Adam_e100_lr0.001/val_n5.svg) | ![val_n5](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n5.svg) |![val_n5](20200420_184537_normal_unet_e100_lr1e-05/val_n5.svg)
große Schwankung | best Wahl | nicht konvergiert(dauert zu lange)

---

* Vergleichen von verschiedenen Modelle

original unet | multiout unet | fcn restnet50 | deeplabv3 resnet50
------------ | ------------- | ------------- | -------------
![val_n1](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n1.svg) | ![val_n1](20200422_111836_multiout_unet_e100_lr0.0001/val_n1.svg) |![val_n1](20200423_155231_fcn_restnet50_e100_lr0.0001/val_n1.svg) |![val_n1](20200425_074639_deeplabv3_resnet50_e100_lr0.0001/val_n1.svg)
![val_n2](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n2.svg) | ![val_n2](20200422_111836_multiout_unet_e100_lr0.0001/val_n2.svg) |![val_n2](20200423_155231_fcn_restnet50_e100_lr0.0001/val_n2.svg) |![val_n2](20200425_074639_deeplabv3_resnet50_e100_lr0.0001/val_n2.svg)
![val_n3](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n3.svg) | ![val_n3](20200422_111836_multiout_unet_e100_lr0.0001/val_n3.svg) |![val_n3](20200423_155231_fcn_restnet50_e100_lr0.0001/val_n3.svg) |![val_n3](20200425_074639_deeplabv3_resnet50_e100_lr0.0001/val_n3.svg)
![val_n4](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n4.svg) | ![val_n4](20200422_111836_multiout_unet_e100_lr0.0001/val_n4.svg) |![val_n4](20200423_155231_fcn_restnet50_e100_lr0.0001/val_n4.svg) |![val_n4](20200425_074639_deeplabv3_resnet50_e100_lr0.0001/val_n4.svg)
![val_n5](20200418_192904_normal_unet_Adam_e100_lr0.0001/val_n5.svg) | ![val_n5](20200422_111836_multiout_unet_e100_lr0.0001/val_n5.svg) |![val_n5](20200423_155231_fcn_restnet50_e100_lr0.0001/val_n5.svg) |![val_n5](20200425_074639_deeplabv3_resnet50_e100_lr0.0001/val_n5.svg)

Y-Bereich von Restnet sind falsch, ich mache morgen neu. 

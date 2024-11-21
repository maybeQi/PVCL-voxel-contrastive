import h5py
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 读取 HDF5 文件
h5_filename = '"A:\PycharmProjects\CPCL-main\data\BraTS2019\data\BraTS19_TCIA10_282_1.h5"'
with h5py.File(h5_filename, 'r') as h5_file:
    # 假设 NIfTI 数据存储在 '/data' 路径下
    nii_data = h5_file['/data'][:]

# 将读取的 NIfTI 数据写入一个临时文件
temp_nii_filename = 'temp_image.nii'
with open(temp_nii_filename, 'wb') as temp_file:
    temp_file.write(nii_data)

# 使用 nibabel 读取临时 NIfTI 文件
nii_image = nib.load(temp_nii_filename)
image_data = nii_image.get_fdata()

# 展示 NIfTI 图像中的一个切片
slice_index = image_data.shape[2] // 2  # 选择中间的切片
plt.imshow(image_data[:, :, slice_index], cmap='gray')
plt.title('NIfTI Image Slice')
plt.axis('off')
plt.show()

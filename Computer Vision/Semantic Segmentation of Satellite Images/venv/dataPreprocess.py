import gdal

img = gdal.Open('C:/Users/Sai Vyas/Desktop/dlt_32N_07/images/32N-12E-230N_01_21')
img = img.ReadAsArray()
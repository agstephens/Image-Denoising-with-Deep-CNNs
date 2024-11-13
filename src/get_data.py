from glob import glob
import reader
import os
from netCDF4 import Dataset
import plotter
import matplotlib as plt_saver
import matplotlib.pyplot as plt
from datetime import datetime

root_path='/neodc/sentinel3a/data/SLSTR/L1_RBT/2023/06/*/'
path_to_your_directory = "/gws/nopw/j04/cloudcatcher/AI_test/"
save_location= "/home/users/cvcox/python/py_projects/AI/Image-Denoising-with-Deep-CNNs/dataset/sat-examples/S4/" #"/gws/nopw/j04/cloudcatcher/AI_test/"
L1_files = glob(root_path+'S3A_SL_1_RBT____202306*'+'NT_004.zip')
counter = 0
# Now loop over all the files found
for f in range(len(L1_files)):
#for f in range(1):
    file = L1_files[f]
    product_name = file[file.index('S3A_'):file.index('S3A_')+94] + '.SEN3/'
    time_label = product_name[25:27]+':'+product_name[27:29]
    date_label = datetime.strptime(product_name[16:24], "%Y%m%d").strftime("%d-%m-%Y")
    date_time_label = date_label + time_label
    UTC_label = date_time_label

    #Extract the name of the L1 product
    print(file[file.index('S3A_'):file.index('S3A_')+94])
    print(f)
    product_name = file[file.index('S3A_'):file.index('S3A_')+94] + '.SEN3/'
    temp_dir = path_to_your_directory + file[file.index('S3A_'):file.index('S3A_')+94] + '.SEN3/'
    #Make the directory
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    #Get the nc files that we want
    tstr='unzip -p '+ file +' */S4_radiance_an.nc'+ ' >'+temp_dir+'S4_radiance_an.nc'  
    print(tstr)
    os.system(tstr)

    tstr='unzip -p '+ file +' */S4_quality_an.nc'+ ' >'+temp_dir+'S4_quality_an.nc'  
    print(tstr)
    os.system(tstr)
    tstr='unzip -p '+ file +' */geometry_tn.nc'+ ' >'+temp_dir+'geometry_tn.nc'  
    print(tstr)
    os.system(tstr)

    file   = reader.Reader(temp_dir)
    s4 = file.reflectance('S4')
    test = plotter.water_channel(file, 'S4', doPlot=False)
    temp_image_file = '%s%s%stemp_image.png'%(save_location, os.path.sep,product_name[16:31])
    print(temp_image_file)
    plt_saver.image.imsave(temp_image_file, test) 

    tstr='rm -fr '+ temp_dir 
    os.system(tstr)
 


  
  #  exit()
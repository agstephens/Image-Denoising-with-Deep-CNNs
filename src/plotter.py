# 
# Copyright: Imperial College London.
# Licence  : GPLv3
# Created  : August 2018
# May 2020: added option not to show plots.
#
import numpy as np
from numpy import float32, zeros, uint16
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import reader
import cv2
import scipy
from decimal import Decimal
from skimage import exposure
#from pyspectral.radiance_tb_conversion import RadTbConverter
#from pyspectral.rsr_reader import RelativeSpectralResponse
#from pyspectral.solar import (SolarIrradianceSpectrum, TOTAL_IRRADIANCE_SPECTRUM_2000ASTM)
#from pyspectral.near_infrared_reflectance import Calculator
#from earthpy.spatial import bytescale

def radiance(reader, rchannel, gchannel , bchannel, doPlot=True):
    """Fill rgb values for a false colour image with the channels given as strings"""
    r   = reader.radiance(rchannel)
    g   = reader.radiance(gchannel)
    b   = reader.radiance(bchannel)
    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)

    if doPlot:
        p = plt.imshow(rgb)
        plt.show()

    return rgb


def flags(reader, conf, offset, doPlot=True):
    #offset is the bit postion of the mask e.g. offset=0 gives the coastline mask in the confidence word
    im_flags = reader.flag(conf, offset)

    if doPlot:
        p = plt.imshow(im_flags)
        plt.show()
    
    
    return im_flags




def image_tir(reader, rchannel, vchannel, doPlot=True):
    
    rchan = reader.radiance(rchannel)
    
    #Read in a visisble channel so I can change the IR array to be high res like that one
    v = reader.radiance(vchannel)
    
    r=v.copy()
    #Increase size of r to match v
    for dim_i in range(1200):
                for dim_j in range(1500):
                        r[dim_i*2,dim_j*2 ] = rchan[dim_i,dim_j]
                        r[dim_i*2,(dim_j*2)+1 ] = rchan[dim_i,dim_j]
                        r[(dim_i*2)+1,dim_j*2 ] = rchan[dim_i,dim_j]
                        r[(dim_i*2)+1,(dim_j*2)+1 ] = rchan[dim_i,dim_j]

    print('max of r ')
    print(r.max())
    
    if doPlot:
        p = plt.imshow(r, cmap="coolwarm",clim=(250,320))#or hot or cool 
        plt.colorbar()
        plt.show()

    
    return r

def cold_test(reader, rchannel, gchannel, bchannel, irchannel, threshold, doPlot=True):

    irchan = reader.radiance(irchannel)
    v = reader.radiance(rchannel)
    ir=v.copy()
#Increase size of ir to match v
    for dim_i in range(1200):
                for dim_j in range(1500):
                        ir[dim_i*2,dim_j*2 ] = irchan[dim_i,dim_j]
                        ir[dim_i*2,(dim_j*2)+1 ] = irchan[dim_i,dim_j]
                        ir[(dim_i*2)+1,dim_j*2 ] = irchan[dim_i,dim_j]
                        ir[(dim_i*2)+1,(dim_j*2)+1 ] = irchan[dim_i,dim_j]


    """Fill rgb values for a false colour image with the channels given as strings"""
    sr = reader.reflectance(rchannel)
    sg = reader.reflectance(gchannel)
    sb = reader.reflectance(bchannel)
  

    
    
   #If a channel is giving Nan/masked value due to brightness of cloud, then fill with the max value   
    r   = np.where( sr.mask == True, sr.max()  ,sr.data )  
    g   = np.where( sg.mask == True, sg.max()  ,sg.data ) 
    b   = np.where( sb.mask == True, sb.max()  ,sb.data )    
  
    if doPlot:
        p = plt.imshow(r)
        plt.colorbar()
        plt.show()  
    
    
    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()


 
    width = r.shape[1]
    height = r.shape[0]
    img = zeros((height, width, 3), dtype=float32)

                
                
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):           
                if ir[y][x] > threshold:
                    img[y][x][0] = min(1, (g[y][x] + b[y][x] ) / 2)
                else:
                    img[y][x][0]  = 0.0


                if ir[y][x] > threshold:
                    img[y][x][1] = min(1, g[y][x] )
                else:
                    img[y][x][1]  = 0.0 
                    
                    
                if ir[y][x] > threshold:
                    img[y][x][2] = min(1, (g[y][x]  + r[y][x] ) / 2)
                else:
                    img[y][x][2]  = 0.0                   
                    
                #img[y][x][0] = min(1, (g[y][x] + b[y][x] ) / 2) if  img[y][x][0] < threshold else img[y][x][0]  = 1.0
               # img[y][x][1] = min(1, g[y][x]) if  img[y][x][0] < threshold else img[y][x][1]  = 0.0
               # img[y][x][2] = min(1, (g[y][x] + r[y][x] ) / 2) if  img[y][x][0] < threshold else img[y][x][2]  = 0.0
                
    rgb_masked = img

    if doPlot:
        p = plt.imshow(rgb_masked)
        plt.show()
    
    rgb_equalized = exposure.equalize_hist(rgb_masked, nbins=512)
    
    return rgb_equalized

def image_day_rgb(reader, rchannel, gchannel, bchannel, doPlot=True):
    #S3, S7, S8
    

    
    s3 = reader.radiance(rchannel)
    gchan = reader.radiance(gchannel)
    bchan = reader.radiance(bchannel)    
    
    #If a channel is giving Nan/masked value due to brightness of cloud, then fill with the max value   
    s3_mod   = np.where( s3.mask == True, s3.max()  ,s3.data ) 
    
    r=s3_mod
    g=r.copy()
    b=r.copy()
    
    #Increase size of g and b to match r
    for dim_i in range(1200):
        for dim_j in range(1500):
            g[dim_i*2:(dim_i*2)+1,dim_j*2:(dim_j*2)+1 ] = gchan[dim_i,dim_j]
            b[dim_i*2:(dim_i*2)+1,dim_j*2:(dim_j*2)+1 ] = bchan[dim_i,dim_j]
    
    
  
    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()

    return r




def reflectance(reader, rchannel, gchannel, bchannel, doPlot=True):
    """Fill rgb values for a false colour image with the channels given as strings"""
    sr = reader.reflectance(rchannel)
    sg = reader.reflectance(gchannel)
    sb = reader.reflectance(bchannel)
    
   #If a channel is giving Nan/masked value due to brightness of cloud, then fill with the max value   
    r   = np.where( sr.mask == True, sr.max()  ,sr.data )  
    g   = np.where( sg.mask == True, sg.max()  ,sg.data ) 
    b   = np.where( sb.mask == True, sb.max()  ,sb.data )    
    
    
    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)
    
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()
    
    return rgb


def bright_test(reader, rchannel, gchannel , bchannel, threshold, doPlot=True):
    """Fill rgb values for a false colour image with the channels given as strings"""
    sr = reader.reflectance(rchannel)
    sg = reader.reflectance(gchannel)
    sb = reader.reflectance(bchannel)
      
   #If a channel is giving Nan/masked value due to brightness of cloud, then fill with the max value   
    r   = np.where( sr.mask == True, sr.max()  ,sr.data )  
    g   = np.where( sg.mask == True, sg.max()  ,sg.data ) 
    b   = np.where( sb.mask == True, sb.max()  ,sb.data )    
  
    if doPlot:
        p = plt.imshow(r)
        plt.colorbar()
        plt.show()  
    
    
    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()


 
    width = r.shape[1]
    height = r.shape[0]
    img = zeros((height, width, 3), dtype=float32)

                
                
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):           
                if g[y][x] < threshold:
                    img[y][x][0] = min(1, (g[y][x] + b[y][x] ) / 2)
                else:
                    img[y][x][0]  = 0.0


                if g[y][x] < threshold:
                    img[y][x][1] = min(1, g[y][x] )
                else:
                    img[y][x][1]  = 0.0 
                    
                    
                if g[y][x] < threshold:
                    img[y][x][2] = min(1, (g[y][x]  + r[y][x] ) / 2)
                else:
                    img[y][x][2]  = 0.0                   
                    
                #img[y][x][0] = min(1, (g[y][x] + b[y][x] ) / 2) if  img[y][x][0] < threshold else img[y][x][0]  = 1.0
               # img[y][x][1] = min(1, g[y][x]) if  img[y][x][0] < threshold else img[y][x][1]  = 0.0
               # img[y][x][2] = min(1, (g[y][x] + r[y][x] ) / 2) if  img[y][x][0] < threshold else img[y][x][2]  = 0.0
                
    rgb_masked = img

  
    if doPlot:
        p = plt.imshow(rgb_masked)
        plt.show()
    
    rgb_equalized = exposure.equalize_hist(rgb_masked, nbins=512)
    
    return rgb_equalized



def reflectance_averaged(reader, s1channel, s3channel, s5channel, doPlot=True):
    """Fill rgb values for a false colour image with the channels given as strings"""
    s1 = reader.reflectance(s1channel)
    s3 = reader.reflectance(s3channel)
    s5 = reader.reflectance(s5channel)


    s5_mod   = np.where( s5.mask == True, s5.max()  ,s5.data )  
    s3_mod   = np.where( s3.mask == True, s3.max()  ,s3.data )   
    s1_mod   = np.where( s1.mask == True, s1.max()  ,s1.data ) 

    width = s3_mod.shape[1]
    height = s3_mod.shape[0]
    img = zeros((height, width, 3), dtype=float32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):           
                img[y][x][0] = min(1, (s3_mod[y][x] + s5_mod[y][x] ) / 2)
                img[y][x][1] = min(1, s3_mod[y][x] )
                img[y][x][2] = min(1, (s3_mod[y][x]  + s1_mod[y][x] ) / 2)
            

    rgb=img
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()

    #rgb_equalized = exposure.equalize_hist(rgb, nbins=512)

    return rgb
    #return rgb_equalized

def red_channel_image(reader, schannel, cmax, doPlot=True):

    """Fill rgb values for a false colour image with the channels given as strings"""
    s = reader.reflectance(schannel)

    s_mod   = np.where( s.mask == True, s.max()  ,s.data )  

    width = s_mod.shape[1]
    height = s_mod.shape[0]
    img = zeros((height, width, 3), dtype=float32)


    high = 1
    low = 0
    cmin = 0. #default data min 
    ##cmax = 0.1  ##np.max(s_mod)
    cscale = cmax - cmin
    scale = float(high - low) / cscale


    for y in range(img.shape[0]):
        for x in range(img.shape[1]):            

                bytedata = ((s_mod[y][x]) * 1.0 - cmin) * scale 

                img[y][x][0] = bytedata
                img[y][x][1] = 0
                img[y][x][2] = 0


    clipped = np.clip(img,0,1)

    rgb=clipped
    return rgb


def green_channel_image(reader, schannel, cmax, doPlot=True):

    """Fill rgb values for a false colour image with the channels given as strings"""
    s = reader.reflectance(schannel)

    s_mod   = np.where( s.mask == True, s.max()  ,s.data )  

    width = s_mod.shape[1]
    height = s_mod.shape[0]
    img = zeros((height, width, 3), dtype=float32)


    high = 1
    low = 0
    cmin = 0. #default data min 
    if cmax == -1:
        cmax = np.max(s_mod)
    cscale = cmax - cmin
    scale = float(high - low) / cscale


    for y in range(img.shape[0]):
        for x in range(img.shape[1]):            

                bytedata = ((s_mod[y][x]) * 1.0 - cmin) * scale 
                img[y][x][0] = 0
                img[y][x][1] = bytedata
                img[y][x][2] = 0


    clipped = np.clip(img,0,1)

    rgb=clipped
    return rgb


def blue_channel_image(reader, schannel,cmax, doPlot=True):

    """Fill rgb values for a false colour image with the channels given as strings"""
    s = reader.reflectance(schannel)

    s_mod   = np.where( s.mask == True, s.max()  ,s.data )  

    width = s_mod.shape[1]
    height = s_mod.shape[0]
    img = zeros((height, width, 3), dtype=float32)


    high = 1
    low = 0
    cmin = 0. #default data min 
    if cmax == -1:
        cmax = np.max(s_mod)


    cscale = cmax - cmin
    scale = float(high - low) / cscale


    for y in range(img.shape[0]):
        for x in range(img.shape[1]):            

                bytedata = ((s_mod[y][x]) * 1.0 - cmin) * scale 
                img[y][x][0] = 0
                img[y][x][1] = 0
                img[y][x][2] = bytedata



    clipped = np.clip(img,0,1)

    rgb=clipped
    return rgb



def water_channel(reader, s4channel, doPlot=True):

    """Fill rgb values for a false colour image with the channels given as strings"""
    s4 = reader.reflectance(s4channel)
    #sunzen = reader.sza()
    #s4_mod   = s4
    s4_mod   = np.where( s4.mask == True, s4.max(), s4.data)  

    width = s4_mod.shape[1]
    height = s4_mod.shape[0]
    img = zeros((height, width, 3), dtype=float32)

    print("max of S4",np.max(s4_mod))
    print("min of S4",np.min(s4_mod))


    high = 1
    low = 0
    cmin = 0. #default data min 
    cmax = .1
    cscale = cmax - cmin
    scale = float(high - low) / cscale

    #clipped =  s4_mod.clip(0,0.4)
    #img[][][0] =  clipped
    #img[][][1] =  clipped
    #img[][][2] =  clipped

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):            

                bytedata = ((s4_mod[y][x]) * 1.0 - cmin) * scale 
                img[y][x][0] = bytedata
                img[y][x][1] = bytedata
                img[y][x][2] = bytedata


    clipped = np.clip(img,0,1)
    print(np.max(img))
    print(np.min(img))
    print(np.max(clipped))
    print(np.min(clipped))

    rgb=clipped
    return rgb

def cloud_type(reader, s4channel, s2channel, s5channel, doPlot=True):

    """Fill rgb values for a false colour image with the channels given as strings"""
    s4 = reader.reflectance(s4channel)
    s2 = reader.reflectance(s2channel)
    s5 = reader.reflectance(s5channel)
    #sunzen = reader.sza()

    s4_mod   = np.where( s4.mask == True, s4.max()  ,s4.data )  
    s5_mod   = np.where( s5.mask == True, s5.max()  ,s5.data )   
    s2_mod   = np.where( s2.mask == True, s2.max()  ,s2.data ) 

    width = s2_mod.shape[1]
    height = s2_mod.shape[0]
    img = zeros((height, width, 3), dtype=float32)

    #Need to adjust range of values
    high = 1
    low = 0
    cmin = 0. #default data min 

    #For red channel
    cmax_r = 0.1
    cscale_r = cmax_r - cmin
    scale_r = float(high - low) / cscale_r

    #For green and blue channels
    cmax_g = .8
    cscale_g = cmax_g - cmin
    scale_g = float(high - low) / cscale_g



    # Apply range limits for each channel. RGB values must be between 0 and 1
    R = ((s4_mod - cmin) / (cmax_r - cmin))  
    G = ((s2_mod - cmin) / (cmax_g - cmin)) 
    B = ((s5_mod - cmin) / (cmax_g - cmin))  




    p = plt.imshow(R,vmin=0,vmax=1.0, cmap='Reds')
    plt.colorbar()
    plt.savefig( "/gws/nopw/j04/cloudcatcher/subjects/CatchCloudGlobal_cirrus/Red.png")


    p = plt.imshow(G,vmin=0,vmax=1.0, cmap='Greens')
    plt.colorbar()
    plt.savefig( "/gws/nopw/j04/cloudcatcher/subjects/CatchCloudGlobal_cirrus/Green.png")
    p = plt.imshow(B,vmin=0,vmax=1.0, cmap='Blues')
    plt.colorbar()
    plt.savefig( "/gws/nopw/j04/cloudcatcher/subjects/CatchCloudGlobal_cirrus/Blue.png")




    # Apply a gamma correction to the image

    #print(np.power(7., Decimal('0.66666')))
    R = np.power(R, Decimal('1.'))
    G = np.power(G, 1./0.75)
    B = np.power(B, 1./1.)
    rgb  =  R +  G + B
    #rgb = np.dstack([R, G, B])
    #for y in range(img.shape[0]):
    #    for x in range(img.shape[1]):           
    #            
    #             img[y][x][0] = ((s4_mod[y][x]) * 1.0 - cmin) * scale_r  
    #             img[y][x][1] = ((s2_mod[y][x]) * 1.0 - cmin) * scale_g    #min(1, s2_mod[y][x])
    #             img[y][x][2] = ((s5_mod[y][x]) * 1.0 - cmin) * scale_g   ##min(1, s5_mod[y][x])
    #            #print(s4_mod[y][x])
    #            #print(((s4_mod[y][x] - cmin) / (cmax_r - cmin)))
    #            img[y][x][0] = ((s4_mod[y][x] - cmin) / (cmax_r - cmin))#**(1./1.5) 
    #            img[y][x][1] = ((s2_mod[y][x] - cmin) / (cmax_g - cmin))#**(1./0.75) 
    #            img[y][x][2] = ((s5_mod[y][x] - cmin) / (cmax_g - cmin))#**(1./1.) 

               
    rgb=img
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()   

    #clipped = np.clip(rgb,0,1)

    rgb[:,:,0]   = np.where( s2.mask == True, float("nan")  ,rgb[:,:,0] )  
    rgb[:,:,1]   = np.where( s2.mask == True, float("nan")  ,rgb[:,:,1] ) 
    rgb[:,:,2]   = np.where( s2.mask == True, float("nan")  ,rgb[:,:,2] ) 
    print("here")
    return rgb


def false_colour(reader, s3channel, s2channel, s1channel, doPlot=True):

    """Fill rgb values for a false colour image with the channels given as strings"""
    s3 = reader.reflectance(s3channel)
    s2 = reader.reflectance(s2channel)
    s1 = reader.reflectance(s1channel)
    #sunzen = reader.sza()

    s3_mod   = np.where( s3.mask == True, s3.max()  ,s3.data )  
    s2_mod   = np.where( s2.mask == True, s2.max()  ,s2.data )   
    s1_mod   = np.where( s1.mask == True, s1.max()  ,s1.data ) 

    width = s2_mod.shape[1]
    height = s2_mod.shape[0]
    img = zeros((height, width, 3), dtype=float32)


    for y in range(img.shape[0]):
        for x in range(img.shape[1]):           
            
                img[y][x][0] = min(1, s3_mod[y][x] )
                img[y][x][1] = min(1, s2_mod[y][x])
                img[y][x][2] = min(1, s1_mod[y][x])

                
    rgb=img
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()   

    rgb[:,:,0]   = np.where( s1.mask == True, float("nan")  ,rgb[:,:,0] )  
    rgb[:,:,1]   = np.where( s1.mask == True, float("nan")  ,rgb[:,:,1] ) 
    rgb[:,:,2]   = np.where( s1.mask == True, float("nan")  ,rgb[:,:,2] ) 
    return rgb



def nat_colour(reader, s5channel, s3channel, s2channel, doPlot=True):

    """Fill rgb values for a false colour image with the channels given as strings"""
    s3 = reader.reflectance(s3channel)
    s2 = reader.reflectance(s2channel)
    s5 = reader.reflectance(s5channel)
    #sunzen = reader.sza()

    s3_mod   = np.where( s3.mask == True, s3.max()  ,s3.data )  
    s2_mod   = np.where( s2.mask == True, s2.max()  ,s2.data )   
    s5_mod   = np.where( s5.mask == True, s5.max()  ,s5.data ) 

    width = s2_mod.shape[1]
    height = s2_mod.shape[0]
    img = zeros((height, width, 3), dtype=float32)


    for y in range(img.shape[0]):
        for x in range(img.shape[1]):           
            
                img[y][x][0] = min(1, s5_mod[y][x] )
                img[y][x][1] = min(1, s3_mod[y][x])
                img[y][x][2] = min(1, s2_mod[y][x])

                
    rgb=img
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()   

    rgb[:,:,0]   = np.where( s2.mask == True, float("nan")  ,rgb[:,:,0] )  
    rgb[:,:,1]   = np.where( s2.mask == True, float("nan")  ,rgb[:,:,1] ) 
    rgb[:,:,2]   = np.where( s2.mask == True, float("nan")  ,rgb[:,:,2] ) 
    return rgb



def reflectance_averaged_masked(reader, s1channel, s3channel, s5channel, conf, offset, doPlot=True):
 
    im_flags=reader.flag(conf,offset)
    
    
    """Fill rgb values for a false colour image with the channels given as strings"""
    s1 = reader.reflectance(s1channel)
    s3 = reader.reflectance(s3channel)
    s5 = reader.reflectance(s5channel)


    s5_mod   = np.where( s5.mask == True, s5.max()  ,s5.data )  
    s3_mod   = np.where( s3.mask == True, s3.max()  ,s3.data )   
    s1_mod   = np.where( s1.mask == True, s1.max()  ,s1.data ) 
    
    s5_mod2   = np.where( im_flags == True, 0.0  ,s5_mod )   
    s3_mod2   = np.where( im_flags == True, 0.0  ,s3_mod ) 
    s1_mod2   = np.where( im_flags == True, 0.0  ,s1_mod ) 

    width = s3_mod2.shape[1]
    height = s3_mod2.shape[0]
    img = zeros((height, width, 3), dtype=float32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):           
                img[y][x][0] = min(1, (s3_mod2[y][x] + s5_mod2[y][x] ) / 2)
                img[y][x][1] = min(1, s3_mod2[y][x] )
                img[y][x][2] = min(1, (s3_mod2[y][x]  + s1_mod2[y][x] ) / 2)
            


    rgb=img
    
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()

    #rgb_equalized = exposure.equalize_hist(rgb, nbins=512)

#return rgb_equalized
    return rgb


def day_microphysics(reader, s8channel, s7channel, s1channel, doPlot=True):

    """Fill rgb values for a false colour image with the channels given as strings"""
    s8 = reader.radiance(s8channel)
    s7 = reader.radiance(s7channel)
    s1 = reader.reflectance(s1channel)
    sunzen = reader.sza()


    s8_mod   = np.where( s8.mask == True, s8.max()  ,s8.data )  
    s7_mod   = np.where( s7.mask == True, s7.max()  ,s7.data )   
    s2_mod   = np.where( s1.mask == True, s1.max()  ,s1.data ) 

    width = s2_mod.shape[1]
    height = s2_mod.shape[0]
    img = zeros((height, width, 3), dtype=float32)


    #Decrease the size of sunzen to be on the in grid
    sunzen_rebin = sunzen[::2, ::2]


    #p = plt.imshow(sunzen_rebin)
    #plt.title('SZA rebin')
    #plt.colorbar()
    #plt.show()



    width_s8 = s8_mod.shape[1]
    height_s8 = s8_mod.shape[0]

    #sunz = np.empty([height_s8,width_s8])
    #sunz.fill(68.0)


 
    #For this recipe, we need the reflective part of S7
    #Following the info on https://pyspectral.readthedocs.io/en/latest/37_reflectance.html
    #convert BT to spectral radiance
    #viirs = RadTbConverter('Sentinel-3A', 'slstr', 'S7')
    refl_s7 = Calculator('Sentinel-3A', 'slstr', 'S7')
    s7r = refl_s7.reflectance_from_tbs(sunzen_rebin, s7_mod, s8_mod)


    #Increase size of S8 and s7r to match S2
    #https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    s8_rebin = s8_mod.repeat(2, axis=0).repeat(2, axis=1)
    s7r_rebin = s7r.repeat(2, axis=0).repeat(2, axis=1)


    #p = plt.imshow(s8_rebin,label='S8')
    #plt.title('S8 rebin')
    #plt.colorbar()
    #plt.show()
      
    

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):           
            
                img[y][x][0] = min(1, (s2_mod[y][x] ) )
                img[y][x][1] = min(0.6, s7r_rebin[y][x] )
                img[y][x][2] = min(1, s8_rebin[y][x]/323.)
                img[y][x][2] = min(0, s8_rebin[y][x]/120.)

                

    rgb=img
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()
   
    #p = plt.imshow(rgb,norm=colors.PowerNorm(gamma=2.5))
    #plt.show()
   
    #rgb_equalized = exposure.equalize_hist(rgb, nbins=512)
    #return (rgb*255).astype(np.uint8)

    rgb[:,:,0]   = np.where( s1.mask == True, float("nan")  ,rgb[:,:,0] )  
    rgb[:,:,1]   = np.where( s1.mask == True, float("nan")  ,rgb[:,:,1] ) 
    rgb[:,:,2]   = np.where( s1.mask == True, float("nan")  ,rgb[:,:,2] ) 
    return rgb


def reflectance_averaged_masked(reader, s1channel, s3channel, s5channel, conf, offset, doPlot=True):
 
    im_flags=reader.flag(conf,offset)
    
    
    """Fill rgb values for a false colour image with the channels given as strings"""
    s1 = reader.reflectance(s1channel)
    s3 = reader.reflectance(s3channel)
    s5 = reader.reflectance(s5channel)


    s5_mod   = np.where( s5.mask == True, s5.max()  ,s5.data )  
    s3_mod   = np.where( s3.mask == True, s3.max()  ,s3.data )   
    s1_mod   = np.where( s1.mask == True, s1.max()  ,s1.data ) 
    
    s5_mod2   = np.where( im_flags == True, 0.0  ,s5_mod )   
    s3_mod2   = np.where( im_flags == True, 0.0  ,s3_mod ) 
    s1_mod2   = np.where( im_flags == True, 0.0  ,s1_mod ) 

    width = s3_mod2.shape[1]
    height = s3_mod2.shape[0]
    img = zeros((height, width, 3), dtype=float32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):           
                img[y][x][0] = min(1, (s3_mod2[y][x] + s5_mod2[y][x] ) / 2)
                img[y][x][1] = min(1, s3_mod2[y][x] )
                img[y][x][2] = min(1, (s3_mod2[y][x]  + s1_mod2[y][x] ) / 2)
            


    rgb=img
    
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()

    #rgb_equalized = exposure.equalize_hist(rgb, nbins=512)

#return rgb_equalized
    return rgb

#This produces an image where the blueish colours are more reddish
def reflectance_enhance(reader, s1channel, s2channel , s3channel, s5channel, doPlot=True):
    """Fill rgb values for a false colour image with the channels given as strings"""
    s1 = reader.reflectance(s1channel)
    s2 = reader.reflectance(s2channel)
    s3 = reader.reflectance(s3channel)
    s5 = reader.reflectance(s5channel)
    
    if doPlot: 
        p = plt.imshow(s1.mask)
        plt.title('s1 mask')
        plt.show()  
  
        p = plt.imshow(s2.mask)
        plt.title('s2 mask')
        plt.show()  
       
        p = plt.imshow(s3.mask)
        plt.title('s3 mask')
        plt.show()  
    
        p = plt.imshow(s5.mask)
        plt.title('s5 mask')
        plt.show()     
    
    
    
    #If a channel is giving Nan/masked value due to brightness of cloud, then fill with the max value   
    s5_mod   = np.where( s5.mask == True, s5.max()  ,s5.data )  
    s3_mod   = np.where( s3.mask == True, s3.max()  ,s3.data )  
    s2_mod   = np.where( s2.mask == True, s2.max()  ,s2.data ) 
    s1_mod   = np.where( s1.mask == True, s1.max()  ,s1.data )    

      
    #calculate ndsi
    ndsi = s5_mod - s1_mod 

    if doPlot:
        p = plt.imshow(ndsi)
        plt.title('ndsi')
        plt.colorbar()
        plt.show()  
    
    
    #Swap bands used for rgb based upon ndsi value
    #Where ndsi>0, use bands S5, S3, S2 (green veg land, blueish clouds)
    #Where ndsi<0, use bands S3, S2, S1 (brown veg land, whiteish clouds)
    r    = np.ma.where(ndsi > -0.15, s5_mod, s3_mod)
    g    = np.ma.where(ndsi > -0.15, s3_mod, s2_mod)
    b    = np.ma.where(ndsi > -0.15, s2_mod, s1_mod)

    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)
    if doPlot:
        p = plt.imshow(rgb)
        plt.show()

    return rgb

    
def snow(reader):
    """Fill rgb values for colour scheme highlighting snow"""
    ndsi = reader.radiance('S5') - reader.radiance('S1')
    r = np.ma.where(ndsi.data > 0, reader.radiance('S5'), reader.radiance('S3'))
    g = np.ma.where(ndsi.data > 0, reader.radiance('S3'), reader.radiance('S2'))
    b = np.ma.where(ndsi.data > 0, reader.radiance('S2'), reader.radiance('S1'))
    rgb = np.ma.dstack((r / r.max(), g / g.max(), b / b.max())).filled(0)
    p = plt.imshow(rgb)
    plt.show()
    
def vis(reader):
    """Fill rgb values for false colour image making ice clouds blue and liquid clouds white/pink"""
    radiance(R, 'S1', 'S2', 'S3')

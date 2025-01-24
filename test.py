for i in range(1, 4):
    if i == 1:
        band = '2GHz'
    elif i == 2:
        band = '2_5GHz'
    elif i == 3:
        band = '3GHz'
    for j in range(7, 10):
        folderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/FMCW radar-based multi-person vital sign monitoring data/2_SymmetricalPosition/1_Radar_Raw_Data/position_ (" + str(j) + ")" #访问三个文件夹
        for k in range(1, 7):
            filePath = r'adc_'+ band +'_position'+ str(j) +'_ (' + str(k) +').bin'
            binPath = folderPath + '/' + filePath
            print(binPath)




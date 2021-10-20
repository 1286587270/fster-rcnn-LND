import os


import pandas
import json

file_dir = "F:/rcnn/luna16/subset4/subset4"
# 获得文件完整路径
list_file = open('data.txt','w', encoding='utf-8')
image_id = []
file1 = []
o = []
for root, dirs, files in os.walk(file_dir):
   for name in files:

     file=os.path.join(root, name)  # 文件

     if  file[-1] == 'd':
         print(file)
         file1.append(file)
         image_id.append(file[31:-4])
         o.append(1)

         #list_file.write('\n')
#print(o)


df = pandas.read_csv('F:/rcnn/luna16/CSVFILES/annotations.csv')
#print(df["seriesuid"][1])
print(len(image_id))
for j in range (0,len(image_id)):
    #
    #print(image_id[j])
    for i in range(0, 1186):

        if (image_id[j] == df["seriesuid"][i]):
            if(o[j]==1):
               list_file.write(file1[j])
               o[j]=0
            list_file.write(" ")
            x,y,z,d= df["coordX"][i],df["coordY"][i],df["coordZ"][i],df["diameter_mm"][i]
            x = str(x)
            y = str(y)
            z = str(z)
            d = str(d)

            list_file.write(x+",")
            list_file.write(y+",")
            list_file.write(z+",")
            list_file.write(d)
            print(x,y,z)
            if(df["seriesuid"][i]!=df["seriesuid"][i+1]):
                list_file.write('\n')





            #print(image_id[j])

list_file.close()



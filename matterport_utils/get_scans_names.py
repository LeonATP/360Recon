# 导入os模块
import os
 
# path定义要获取的文件名称的目录
path = "/home/mu/sda1/dataset/output_directory/v1/Matterport3D"
output_path = "/home/mu/sda1/simplerecon/data_splits/Matterport3d/"
 
# os.listdir()方法获取文件夹名字，返回数组
file_name_list = os.listdir(path)
 
# 转为转为字符串
file_name = str(file_name_list)
 
# replace替换"["、"]"、" "、"'"
file_name = file_name.replace("[", "").replace("]", "").replace("'", "").replace(",", "\n").replace(" ", "")
 
# 创建并打开文件list.txt
f = open(output_path + "matterport3d.txt", "w")
 
# 将文件下名称写入到"文件list.txt"
f.write(file_name)
import os
import re

def clean_filename(filename):
    """
    去除文件名中的符号和括号，只保留中英文、数字和空格
    """
    # 保留中文字符、英文字母、数字和空格
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9\s]'
    cleaned = re.sub(pattern, '', filename)
    return cleaned.strip()

def rename_files_in_folder(folder_path):
    """
    重命名文件夹下的所有文件
    """
    for filename in os.listdir(folder_path):
        # 获取文件完整路径
        old_path = os.path.join(folder_path, filename)
        
        # 跳过目录
        if os.path.isdir(old_path):
            continue
            
        # 分离文件名和扩展名
        basename, ext = os.path.splitext(filename)
        
        # 清理文件名
        new_basename = clean_filename(basename)
        new_filename = f"{new_basename}{ext}"
        new_path = os.path.join(folder_path, new_filename)
        
        # 避免重名冲突
        counter = 1
        while os.path.exists(new_path) and new_path != old_path:
            new_filename = f"{new_basename}_{counter}{ext}"
            new_path = os.path.join(folder_path, new_filename)
            counter += 1
            
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_filename}")

if __name__ == "__main__":
    folder = input("请输入文件夹路径: ")
    if os.path.isdir(folder):
        rename_files_in_folder(folder)
        print("所有文件处理完成!")
    else:
        print("错误: 指定的路径不是文件夹")

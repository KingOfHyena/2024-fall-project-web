import traceback

from flask import Flask, request, jsonify, send_from_directory
from celery_tasks import generate_heatmap
import os
import torch
from flask_cors import CORS  # 导入CORS
import pandas as pd
from dataset import MyDataset
import numpy as np
import matplotlib.pyplot as plt
from model import MyCNN
app = Flask(__name__)
CORS(app)  # 启用CORS，允许所有域名的请求

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
UPLOAD_FOLDER = './static/uploads/'
HEATMAP_FOLDER = './static/heatmap/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER

# 标签列表及对应名称
labels = {
    1: '仰卧',
    2: '俯卧',
    3: '左侧卧（弯腿）',
    4: '左侧卧（伸直）',
    5: '右侧卧（弯腿）',
    6: '右侧卧（伸直）',
    7: '坐'
}

model = MyCNN()
model = torch.load('./model/cnn_10_750_1.0.model', map_location=torch.device('cpu'))
model.eval()
def plot_heatmap(data, output_file):
    plt.imshow(data, cmap='turbo', interpolation='bilinear')
    plt.colorbar()
    plt.savefig(output_file)
    plt.close()


def generate_heatmap(filepath):
    output_dir = './static/heatmap/'  # 热力图保存目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取 Excel 文件
    xlsx_data = pd.read_excel(filepath, sheet_name=None)

    heatmap_paths = []  # 用于存储生成的热力图路径

    # 遍历每张表
    for sheet_name, df in xlsx_data.items():
        if sheet_name == '力值':
            data_force = df.loc[0:, '(0,0)':'(25,39)'].values  # 提取除了第一行以外的数据

            for i, row in enumerate(data_force):
                row_two = row.reshape(40, 26, order='F')  # 将数据重塑为 40x26 的数组
                output_file = os.path.join(output_dir, f"heatmap_{i}.png")
                plot_heatmap(row_two, output_file)
                heatmap_paths.append(output_file)  # 将生成的热力图路径添加到列表中

    return heatmap_paths  # 返回所有生成的热力图路径


@app.route('/upload', methods=['POST'])
def upload_file():
    print("upload_file 函数被调用")  # 确认函数是否被调用的打印语句

    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if not file.filename.endswith('.xlsx'):
        return jsonify({'error': '文件格式不支持，仅支持.xlsx文件'}), 400

    filepath = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), file.filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'文件保存失败: {str(e)}'}), 500

    try:
        xlsx_data = pd.read_excel(filepath, sheet_name='力值')
        print("DataFrame 内容:\n", xlsx_data)  # 打印原始DataFrame

        # 提取表头
        headers = xlsx_data.columns.tolist()

        # 将每一行的数据转化为列表
        row_data = xlsx_data.values.tolist()

        # 返回前40行的简要信息
        summary_rows = [
            {"index": i, "summary": {headers[j]: value for j, value in enumerate(row[:5])}}
            for i, row in enumerate(row_data[:40])  # 这里限制为返回前40行，你可以根据需求调整
        ]

        # 打印生成的摘要信息
        print("生成的摘要信息:\n", summary_rows)

        return jsonify({
            'message': '文件上传成功',
            'rows': summary_rows,  # 返回的rows现在是摘要信息
            'total_rows': len(row_data),  # 返回总行数，以便前端进行分页处理
            'headers': headers,  # 返回表头信息
            'file_path':filepath
        }), 200

    except Exception as e:
        return jsonify({'error': f'解析文件失败: {str(e)}'}), 500


@app.route('/run_model', methods=['POST'])
def run_model():
    try:
        data = request.get_json()
        selected_row_index = data.get('selected_row')
        file_path = data.get('file_path')

        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': '文件路径无效或文件不存在'}), 400

        # 读取 Excel 文件
        xlsx_data = pd.read_excel(file_path, sheet_name=None)

        data_force = None

        # 遍历每张表

        for sheet_name, df in xlsx_data.items():
            if sheet_name == '力值':
                data_force = df.loc[0:, '(0,0)':'(25,39)'].values  # 提取数据

        if data_force is None:
            return jsonify({'error': '未找到名为 "力值" 的表格或数据为空'}), 400

        # 选取指定行的数据并reshape为40x26的格式
        selected_row = data_force[selected_row_index].reshape(40, 26, order='F')

        # 使用数据生成热力图并进行预测
        output_file = f'static/heatmap/heatmap_{selected_row_index}.png'
        print(output_file)
        plot_heatmap(selected_row, output_file)
        selected_row = selected_row[np.newaxis, np.newaxis, :, :]  # 变为 [1, 1, 40, 26] 的形状
        # 将数据转换为Tensor
        tensor_data = torch.tensor(selected_row, dtype=torch.float32)
        # 使用模型进行预测
        with torch.no_grad():
            output = model(tensor_data)
            _, predicted = torch.max(output, 1)
            predicted_label = labels[int(predicted)]

        print("hello")
        print(f"标签是{predicted_label}")
        return jsonify({
            'heatmap_path': output_file,
            'predicted_label': predicted_label
        }), 200
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("fuckyou")
        traceback.print_exc()  # 打印完整的错误堆栈
        return jsonify({'error': f'处理数据时出错: {str(e)}'}), 500


@app.route('/heatmap/<filename>')
def get_heatmap(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(HEATMAP_FOLDER, exist_ok=True)
    app.run(debug=True)

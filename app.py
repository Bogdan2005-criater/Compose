from flask import Flask, render_template, request, jsonify
import os
from composite_predictor import CompositeModelPredictor

app = Flask(__name__)

predictor = CompositeModelPredictor(model_dir='models')

DEFAULT_VALUES = {
    'Угол нашивки, град': 0,
    'Шаг нашивки': 4,
    'Плотность нашивки': 57,
    'Соотношение матрица-наполнитель': 1.85714285714285,
    'Плотность, кг/м3': 2030,
    'модуль упругости, ГПа': 738.736842105263,
    'Количество отвердителя, м.%': 30,
    'Содержание эпоксидных групп,%_2': 22.2678571428571,
    'Температура вспышки, С_2': 100,
    'Поверхностная плотность, г/м2': 210,
    'Потребление смолы, г/м2': 220
}

@app.route('/')
def index():
    return render_template('index.html', default_values=DEFAULT_VALUES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {}
        for key in DEFAULT_VALUES.keys():
            value = request.form.get(key)
            input_data[key] = float(value) if value and value.strip() != '' else DEFAULT_VALUES[key]
        

        properties, ratio = predictor.predict_optimal_composite(input_data)
        
        results = {
            'properties': {
                'Модуль упругости при растяжении, ГПа': round(properties['Модуль упругости при растяжении, ГПа'], 2),
                'Прочность при растяжении, МПа': round(properties['Прочность при растяжении, МПа'], 2)
            },
            'ratio': round(ratio['Соотношение матрица-наполнитель'], 4),
            'success': True
        }
    except Exception as e:
        results = {
            'error': str(e),
            'success': False
        }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

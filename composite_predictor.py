import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import os

class CompositeModelPredictor:
    def __init__(self, model_dir='models'):
        """
        Инициализация предсказателя композитных материалов
        
        Args:
            model_dir (str): Директория с файлами моделей и масштабировщиков
        """
        self.model_dir = model_dir
        self.composite_model = None
        self.composite_scaler_X = None
        self.composite_scaler_y = None
        self.composite_features = None
        
        self.ratio_model = None
        self.ratio_scaler = None
        self.ratio_features = None
        
        self.load_models()
    
    def load_models(self):
        """Загрузка всех моделей и вспомогательных объектов"""
        try:
            self.composite_model = load_model(os.path.join(self.model_dir, 'composite_properties_model.keras'))
            self.composite_scaler_X = joblib.load(os.path.join(self.model_dir, 'composite_scaler_X.pkl'))
            self.composite_scaler_y = joblib.load(os.path.join(self.model_dir, 'composite_scaler_y.pkl'))
            self.composite_features = joblib.load(os.path.join(self.model_dir, 'composite_features.pkl'))
            
            self.ratio_model = load_model(os.path.join(self.model_dir, 'matrix_ratio_model.keras'))
            self.ratio_scaler = joblib.load(os.path.join(self.model_dir, 'matrix_ratio_scaler.pkl'))
            self.ratio_features = joblib.load(os.path.join(self.model_dir, 'matrix_ratio_features.pkl'))
            
            print("Все модели успешно загружены")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки моделей: {str(e)}")
    
    def predict_composite_properties(self, input_data):
        """
        Предсказание механических свойств композита
        
        Args:
            input_data (dict): Словарь с входными параметрами
        
        Returns:
            dict: Предсказанные свойства
        """
        if not self.composite_model:
            raise RuntimeError("Модель для свойств композита не загружена")
        
        input_df = pd.DataFrame([input_data])
        

        missing_features = [f for f in self.composite_features if f not in input_df.columns]
        if missing_features:
            raise ValueError(f"Отсутствуют обязательные признаки: {', '.join(missing_features)}")
        
        input_df = input_df[self.composite_features]
        scaled_input = self.composite_scaler_X.transform(input_df)
        
        scaled_predictions = self.composite_model.predict(scaled_input)
        predictions = self.composite_scaler_y.inverse_transform(scaled_predictions)
        
        return {
            'Модуль упругости при растяжении, ГПа': float(predictions[0][0]),
            'Прочность при растяжении, МПа': float(predictions[0][1])
        }
    
    def predict_matrix_ratio(self, input_data):
        """
        Предсказание оптимального соотношения матрица-наполнитель
        
        Args:
            input_data (dict): Словарь с входными параметрами
        
        Returns:
            dict: Предсказанное соотношение
        """
        if not self.ratio_model:
            raise RuntimeError("Модель для соотношения матрица-наполнитель не загружена")
        
        input_df = pd.DataFrame([input_data])
        
        missing_features = [f for f in self.ratio_features if f not in input_df.columns]
        if missing_features:
            raise ValueError(f"Отсутствуют обязательные признаки: {', '.join(missing_features)}")
        
        input_df = input_df[self.ratio_features]
        scaled_input = self.ratio_scaler.transform(input_df)
        
        prediction = self.ratio_model.predict(scaled_input)[0][0]
        return {'Соотношение матрица-наполнитель': float(prediction)}
    
    def predict_optimal_composite(self, input_data):
        """
        Полный цикл предсказания: сначала свойства, затем соотношение
        
        Args:
            input_data (dict): Словарь с входными параметрами
        
        Returns:
            tuple: (свойства композита, соотношение матрица-наполнитель)
        """
        properties = self.predict_composite_properties(input_data)
        
        ratio_input = {**input_data, **properties}
        ratio = self.predict_matrix_ratio(ratio_input)
        
        return properties, ratio


if __name__ == "__main__":
    predictor = CompositeModelPredictor()
    
    composite_input = {
        'Угол нашивки, град': 1,
        'Шаг нашивки': 12,
        'Плотность нашивки': 45,
        'Соотношение матрица-наполнитель': 0.75,
        'Плотность, кг/м3': 1750,
        'модуль упругости, ГПа': 22.5,
        'Количество отвердителя, м.%': 27,
        'Содержание эпоксидных групп,%_2': 15,
        'Температура вспышки, С_2': 250,
        'Поверхностная плотность, г/м2': 300,
        'Потребление смолы, г/м2': 120
    }
    

    try:
        composite_result = predictor.predict_composite_properties(composite_input)
        print("\nПредсказанные свойства композита:")
        for prop, value in composite_result.items():
            print(f"{prop}: {value:.2f}")
    except Exception as e:
        print(f"Ошибка предсказания свойств: {str(e)}")
    
    matrix_ratio_input = {
        'Угол нашивки, град': 1,
        'Шаг нашивки': 12,
        'Плотность нашивки': 45,
        'Плотность, кг/м3': 1750,
        'модуль упругости, ГПа': 22.5,
        'Количество отвердителя, м.%': 27,
        'Содержание эпоксидных групп,%_2': 15,
        'Температура вспышки, С_2': 250,
        'Поверхностная плотность, г/м2': 300,
        'Потребление смолы, г/м2': 120,
        'Модуль упругости при растяжении, ГПа': 120,
        'Прочность при растяжении, МПа': 2500
    }
    
    try:
        ratio_result = predictor.predict_matrix_ratio(matrix_ratio_input)
        print("\nПредсказанное соотношение матрица-наполнитель:")
        print(f"{ratio_result['Соотношение матрица-наполнитель']:.4f}")
    except Exception as e:
        print(f"Ошибка предсказания соотношения: {str(e)}")
    
    try:
        properties, ratio = predictor.predict_optimal_composite(composite_input)
        print("\nРезультат полного цикла предсказания:")
        print("Свойства композита:")
        for prop, value in properties.items():
            print(f"  {prop}: {value:.2f}")
        print(f"Оптимальное соотношение: {ratio['Соотношение матрица-наполнитель']:.4f}")
    except Exception as e:
        print(f"Ошибка полного цикла предсказания: {str(e)}")

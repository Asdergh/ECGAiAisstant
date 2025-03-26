import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, ecg_data, reports, transform=None):
        """
        Инициализация датасета.

        Аргументы:
            ecg_data (list): Список записей с данными ЭКГ. Каждая запись должна быть словарем с параметрами:
                'rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis'
            reports (list): Список отчетов (строк), соответствующих каждой записи ЭКГ.
            transform (callable, опционально): Функция для преобразования данных ЭКГ.
        """
        assert len(ecg_data) == len(reports), "Количество данных ЭКГ должно совпадать с количеством отчетов."
        self.ecg_data = ecg_data
        self.reports = reports
        self.transform = transform

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg_sample = self.ecg_data[idx]
        report = self.reports[idx]

        # Если задано преобразование, применяем его
        if self.transform:
            ecg_sample = self.transform(ecg_sample)

        # Преобразуем значения параметров в тензор
        keys = ['rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']
        ecg_tensor = torch.tensor([ecg_sample[key] for key in keys], dtype=torch.float)

        return {"ecg": ecg_tensor, "report": report}


if __name__ == '__main__':
    # Имитация данных ЭКГ с параметрами QRS
    ecg_samples = [
        {
            'rr_interval': 0.8,
            'p_onset': 0.1,
            'p_end': 0.2,
            'qrs_onset': 0.3,
            'qrs_end': 0.4,
            't_end': 0.6,
            'p_axis': 45.0,
            'qrs_axis': 60.0,
            't_axis': 70.0,
        },
        {
            'rr_interval': 0.9,
            'p_onset': 0.15,
            'p_end': 0.25,
            'qrs_onset': 0.35,
            'qrs_end': 0.45,
            't_end': 0.65,
            'p_axis': 50.0,
            'qrs_axis': 65.0,
            't_axis': 75.0,
        }
    ]

    # Соответствующие отчеты
    reports = [
        "Отчет для примера 1: параметры в норме",
        "Отчет для примера 2: небольшое отклонение",
    ]

    dataset = ECGDataset(ecg_samples, reports)
    sample = dataset[0]
    print("ECG данные:", sample["ecg"])
    print("Отчет:", sample["report"])
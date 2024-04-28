import openpyxl
import string

def impoport(name):
    # Открываем файл Excel
    workbook = openpyxl.load_workbook(name)
    # Выбираем активный лист (первый лист в книге)
    sheet = workbook.active

    # Создаем список для хранения текста из каждой ячейки
    cell_text_list = []

    # Проходим по каждой строке и столбцу в таблице
    for row in sheet.iter_rows(values_only=True):
        for cell_value in row:
            if cell_value is not None:
                cell_text_list.append(str(cell_value))
    # Закрываем файл Excel
    workbook.close()
    return cell_text_list

def remove_punctuation(text):
    # Удаляем знаки препинания из текста
    cleaned_text = ''.join(char for char in text if char not in string.punctuation).lower()
    return cleaned_text


allrev = "allrevs.xlsx"
checkrev = "chekList.xlsx"
datawords = "уникальные_слова.xlsx"

# Массивы с отзывами и словами для датасета
check_revs = impoport(checkrev)
dataset_words = impoport(datawords)
all_revs = impoport(allrev)


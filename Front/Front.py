import platform
import subprocess
from typing import Dict
import os
import shutil
import flet as ft
import sys
from flet import *
from Neuron import superAI
from FunctionalAPI import fAPI_test
from FastText import FT_test
from Sequentinal import Sequentinal_model
from word2vec import W2V_test
from Excel import count_voices_excel

global input_text_fromFront
global nets_votes, nets_votes_excel

options = ["FunctionalAPI", "Sequentinal", "Neuron", "FastText", "Word2Vec"]
ALL_runned = False
nets_votes = 0


def main(page: ft.Page):
    BG = '#1B3A50'
    VIOLET = '#63359E'
    FG = '#3450a1'
    PINK = '#eb06ff'
    page.theme = Theme(font_family="Consolas")
    page.title = 'ToneApp'
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    output_text = ft.Text()
    review_from_front = ft.TextField(label="Введите отзыв",
                                     autofocus=True,
                                     multiline=True,
                                     max_lines=3,
                                     color=ft.colors.BLUE_GREY_600,
                                     focused_border_color=ft.colors.BLUE_GREY_600,
                                     focused_color=ft.colors.BLUE_GREY_600)
    answer = ft.Column()
    detailed_answer = ft.Column()

    # функция для удаления предыдущих предиктов
    def clear_prediction():
        answer.controls = []
        detailed_answer.controls = []
        page.update()

    def run_neuron():
        global nets_votes
        if not ALL_runned: clear_prediction()
        input_text_fromFront = review_from_front.value
        if superAI.for_front(input_text_fromFront) == 0:  # Correctly reference the function from superAI
            detailed_answer.controls.append(ft.Text("Neuron: -", size=10))
            nets_votes -= 1
        else:
            detailed_answer.controls.append(ft.Text("Neuron: +", size=10))
            nets_votes += 1
        page.update()
        review_from_front.focus()

    def run_FunctionalAPI():
        global nets_votes
        if not ALL_runned: clear_prediction()
        input_text_fromFront = review_from_front.value
        if fAPI_test.for_front_Funcstional(input_text_fromFront) == 0:
            detailed_answer.controls.append(ft.Text("Functional: -", size=10))
            nets_votes -= 1
        else:
            detailed_answer.controls.append(ft.Text("Functional: +", size=10))
            nets_votes += 1
        page.update()
        review_from_front.focus()

    def run_FastText():
        global nets_votes
        if not ALL_runned: clear_prediction()
        input_text_fromFront = review_from_front.value
        print(input_text_fromFront)
        if FT_test.FastText(input_text_fromFront) == 0:
            detailed_answer.controls.append(ft.Text("FastText: -", size=10))
            nets_votes -= 1
        else:
            detailed_answer.controls.append(ft.Text("FastText: +", size=10))
            nets_votes += 1
        page.update()
        review_from_front.focus()

    def run_Sequentinal():
        global nets_votes
        if not ALL_runned: clear_prediction()
        input_text_fromFront = review_from_front.value
        print(input_text_fromFront)
        if Sequentinal_model.for_front_Sequentinal(input_text_fromFront) == 0:
            detailed_answer.controls.append(ft.Text("Sequentinal: -", size=10))
            nets_votes -= 1
        else:
            detailed_answer.controls.append(ft.Text("Sequentinal: +", size=10))
            nets_votes += 1
        page.update()
        review_from_front.focus()

    def run_Word2Vec():
        global nets_votes
        if not ALL_runned: clear_prediction()
        input_text_fromFront = review_from_front.value
        print(input_text_fromFront)
        if W2V_test.W2V_test(input_text_fromFront) == 0:
            detailed_answer.controls.append(ft.Text("Word2Vec: -", size=10))
            nets_votes -= 1
        else:
            detailed_answer.controls.append(ft.Text("Word2Vec: +", size=10))
            nets_votes += 1
        page.update()
        review_from_front.focus()

    def run_All():
        global ALL_runned
        global nets_votes
        nets_votes = 0
        ALL_runned = True
        clear_prediction()
        run_FunctionalAPI()
        run_neuron()
        run_FastText()
        run_Sequentinal()
        run_Word2Vec()
        ALL_runned = False
        if nets_votes > 0:
            answer.controls.append(
                ft.Text("ПОЛОЖИТЕЛЬНЫЙ", color=ft.colors.GREEN_800, size=50, weight=ft.FontWeight.W_500))
        else:
            answer.controls.append(ft.Text("ОТРИЦАТЕЛЬНЫЙ", color=ft.colors.DEEP_ORANGE_900, size=50, weight=ft.FontWeight.W_500))
        page.update()

    prog_bars: Dict[str, ProgressRing] = {}
    files = Ref[Column]()
    upload_button = Ref[ElevatedButton]()

    def file_run_ALL(destination):
        superAI.neuron_test_excel(destination)
        fAPI_test.for_front_Functional_excel(destination)
        Sequentinal_model.for_front_Sequentinal_excel(destination)
        FT_test.FastText_excel(destination)
        W2V_test.W2V_text_excel(destination)
        count_voices_excel(destination)

    def file_picker_result(e: FilePickerResultEvent):
        upload_button.current.disabled = True if e.files is None else False
        prog_bars.clear()
        files.current.controls.clear()
        if e.files is not None:
            for f in e.files:
                prog = ProgressRing(value=0, bgcolor="#eeeeee", width=20, height=20)
                prog_bars[f.name] = prog
                files.current.controls.append(Row([prog, Text(f.name, color=ft.colors.BLUE_GREY_600)]))
        page.update()

    def on_upload_progress(e: FilePickerUploadEvent):
        prog_bars[e.file_name].value = e.progress
        prog_bars[e.file_name].update()

    def upload_files(e):
        if file_picker.result is not None and file_picker.result.files is not None:
            for f in file_picker.result.files:
                upload_folder = os.path.join(os.getcwd(), "Uploads")
                destination = os.path.join(upload_folder, os.path.basename(f.path))
                shutil.copy(f.path, destination)
                file_run_ALL(destination)
                page.dialog = post_upload_dlg
                post_upload_dlg.open = True
                page.update()

    # функция для кнопки "определить тональность"
    def btn_click(e):
        planet.visible = False
        greetings.visible = False
        pr.visible = True
        page_2.update()
        run_All()
        pr.visible = False
        info_icon.visible = True
        page_2.update()

    def btn_more_info(e):
        if page_3.height == 150:
            page_3.height = 0
        else:
            page_3.height = 150
        page_3.visible = True
        page_3.update()

    def btn_about_app(e):
        page.dialog = about_app_dlg
        about_app_dlg.open = True
        page.update()

    def open_project_folder(e):
        project_folder = os.getcwd()
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(['explorer', project_folder])
        elif system == "Darwin":  # macOS
            subprocess.Popen(['open', project_folder])
        elif system == "Linux":
            subprocess.Popen(["xdg-open", project_folder])
        else:
            print("Не удалось определить ОС.")

    def exit_app(e):
        print("EXIT!")
        page.window_close()


    ans_btn = ft.ElevatedButton("Определить тональность",
                                on_click=btn_click,
                                height=55,
                                style=ft.ButtonStyle(
                                    color={ft.MaterialState.HOVERED: 'white',
                                           ft.MaterialState.FOCUSED: 'white',
                                           ft.MaterialState.DEFAULT: 'white'
                                           },
                                    bgcolor=ft.colors.BLUE_GREY_600,
                                ))

    file_picker = FilePicker(on_result=file_picker_result, on_upload=on_upload_progress)
    select_files_btn = ft.ElevatedButton(
        "Выбрать Excel-файл...",
        icon=icons.FOLDER_OPEN,
        height=55,
        on_click=lambda _: file_picker.pick_files(allow_multiple=True),
        color='white',
        bgcolor=ft.colors.BLUE_GREY_600
    )

    upload_btn = ft.ElevatedButton(
        "Загрузить",
        ref=upload_button,
        icon=icons.UPLOAD,
        on_click=upload_files,
        disabled=True,
        color='white',
        bgcolor=ft.colors.BLUE_GREY_600
    )

    post_upload_dlg = ft.AlertDialog(
        content=ft.Text("Анализ успешно завершен. Изменения сохранены в новом файле в папке проекта Uploads.",
                        width=400, size=16, color=ft.colors.BLUE_GREY_600),
        actions=[
            ft.TextButton("Открыть",
                          on_click=open_project_folder,
                          style=ft.ButtonStyle(
                              color=ft.colors.BLUE_GREY_600
                          ))
        ],
        on_dismiss=lambda e: print("Modal"),
        bgcolor=ft.colors.GREY_300
    )

    about_app_dlg = ft.AlertDialog(
        content=ft.Text(" Наше приложение представляет собой интеллектуальный анализатор текста, который "
                        "помогает определить тональность текста, анализируя его на наличие позитивного или негативного"
                        " контента. Это полезный инструмент для людей, работающих с большими объемами текста, таких "
                        "как отзывы, комментарии или социальные медиа.\n\n"
                        "Как это работает: \n\n1. Пользователь вводит текст или загружает файл с текстом. \n2. Приложение запускает "
                        "несколько алгоритмов машинного обучения для анализа текста.\n3. Каждый алгоритм возвращает "
                        "оценку тональности текста (положительный или отрицательный).\n4. По результатам всех алгоритмов "
                        "приложение определяет общую тональность текста. \n5. Результаты анализа отображаются на экране, "
                        "позволяя пользователю легко интерпретировать результаты.",
                        width=400, size=16, color=ft.colors.BLUE_GREY_600),
        bgcolor=ft.colors.GREY_300
    )

    page_1 = ft.Column(
        controls=[
            review_from_front,
            ans_btn,
            output_text,
            select_files_btn,
            ft.Column(ref=files),
            upload_btn,
        ]
    )

    # окно ожидания - когда ничего не происходит

    planet = ft.Image(src=f"planet_waiting_icon.png",
                      height=400)

    greetings = ft.Text("Здесь пока пусто...\nНо вы на правильном пути!"
                        "\nОпределите тональность текста в несколько кликов.",
                        color=ft.colors.BLUE_GREY_600,
                        text_align=TextAlign.CENTER,
                        size=18)

    # progress ring
    pr = ft.ProgressRing(width=100, height=100, stroke_width=10, visible=False, color=colors.DEEP_PURPLE_100)

    info_icon = ft.Container(ft.IconButton(
        icon=ft.icons.EXPAND_MORE, on_click=btn_more_info,

    ),
        alignment=ft.alignment.top_right,
        visible=False
    )

    page_3 = ft.Container(ft.Row(
        controls=[
            detailed_answer
        ],
    ),
        width=300,
        height=0,  # 100
        bgcolor=colors.BLUE_GREY_600,
        margin=20,
        padding=20,
        visible=False,
        animate=ft.animation.Animation(300, "easeOutExpo"), # анимация окошка с нейронками
        alignment=alignment.center_right
    )

    page_4 = ft.Stack(
        [
            page_3,
            info_icon
        ],
    )

    page_2 = ft.Column(
        controls=[
            planet,
            greetings,
            pr,
            answer,
            page_4
        ],
        spacing=50,
    )


    TextContainer = ft.Container(
        content=Row(
            controls=[
                page_1,
                page_2
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        ),
        padding=padding.only(
            right=225, left=100
        ),
    )

    page.appbar = ft.AppBar(
        leading=ft.Image(src=f"header.png", opacity=0.5),
        leading_width=300,
        toolbar_height=70,
        bgcolor=ft.colors.BLUE_GREY_600,
        title=ft.Text("Интеллектуальный анализатор v1.0"),
        center_title=False,
        actions=[
            ft.PopupMenuButton(
                items=[
                    ft.PopupMenuItem(
                        text="Как это работает?",
                        on_click=btn_about_app
                    ),
                    ft.PopupMenuItem(
                        text="Выход",
                        on_click=exit_app
                    )
                ]
            )
        ]
    )

    page.bgcolor = ft.colors.GREY_200

    page.overlay.append(file_picker)

    page.add(
        TextContainer
    )


ft.app(target=main, assets_dir="assets", upload_dir="Uploads")
# view=ft.AppView.WEB_BROWSER

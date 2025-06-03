def remove_markdown_tables(input_file, output_file):
    # Открываем файлы для чтения и записи
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        in_table = False  # Флаг нахождения внутри таблицы
        prev_line = ''  # Хранение предыдущей строки

        for line in f_in:
            stripped_line = line.strip()

            # Если находимся внутри таблицы
            if in_table:
                # Проверяем, закончилась ли таблица
                if not stripped_line.startswith('|') or len(stripped_line) == 0:
                    in_table = False
                continue

            # Проверка на начало таблицы: текущая строка содержит |, следующая строка содержит ---
            if line.startswith('|') and '|' in line and '---' in prev_line:
                in_table = True
                prev_line = ''  # Сброс предыдущей строки
                continue

            # Запись строки, если не часть таблицы
            if not in_table:
                f_out.write(line)

            prev_line = stripped_line  # Обновление предыдущей строки


# Использование
# remove_markdown_tables('../../output.md', 'cleaned_output.md')
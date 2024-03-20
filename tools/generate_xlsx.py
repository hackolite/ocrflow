import xlsxwriter

data = [[0,0,0,0,0,0,0,0,True], [0,0,0,0,0,0,0,0,True]]
# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('tmp.xlsx')
worksheet = workbook.add_worksheet()
bool_format = workbook.add_format({'bold': True, 'color': 'red'})


# Create a format for the specific cell
#cell_format = workbook.add_format({'bold': True, 'color': 'red'})
# Write a blank cell with the specified format to the specific cell ('A1')


worksheet.set_column('J:J', None, bool_format)
worksheet.write('A1', 'link')
worksheet.write('B1', 'image')
worksheet.write('C1', 'gtin')
worksheet.write('D1', 'price')
worksheet.write('E1', 'digital_check')
worksheet.write('F1', 'human_check')
worksheet.write('G1', 'pad_check')
worksheet.write('H1', 'text_check')
worksheet.write('I1', 'ean_check')
worksheet.write('J1', 'osm_id')


for row_num, value in enumerate(data):
    # Insert checkbox in each cell of the row
    print(value)
    worksheet.write_row(0, 0, value[:-1])
    worksheet.insert_checkbox(row_num, len(data), {'checked': True})

workbook.close()
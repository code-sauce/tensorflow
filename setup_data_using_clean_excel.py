import solr
from PIL import Image
import requests
from StringIO import StringIO
from xlrd import open_workbook
import os

PROJECT_PATH = os.path.abspath(os.path.split(__file__)[0])
TRAIN_DATA_DIR = os.path.join(PROJECT_PATH, 'data', 'train')
VALIDATION_DATA_DIR = os.path.join(PROJECT_PATH, 'data', 'validate')
CLEAN_EXCEL_DATA_DIR = os.path.join(PROJECT_PATH, 'data', 'excel_clean_data')
train_category_to_image_map = {}
start = 0
BATCH_SIZE = 200
batch_count = 0

LABEL_TO_EXCEL_FILE_MAPPING = {
    'bags': ['backpack.xlsx', 'handbags.xlsx', 'laptop bags.xlsx'],
    'bottoms': ['pants.xlsx', 'shorts.xlsx', 'skirts.xlsx'],
    'dresses': ['dresses.xlsx'],
    'shoes': ['mens boots.xlsx', 'mens oxfords.xlsx', 'mens sandals.xlsx', 'mens sneakers.xlsx',
              'womens boots.xlsx', 'womens flats.xlsx', 'womens pumps.xlsx', 'womens sandals.xlsx',
              'womens sneakers.xlsx'],
    'tops': ['women_teams.xlsx', 'womens_blouses.xlsx', 'womens_tops.xlsx']
}

s = solr.SolrConnection('http://solr-prod.s-9.us:8983/solr/shoprunner')

for training_label, clean_excel_file_names in LABEL_TO_EXCEL_FILE_MAPPING.items():
    print 'getting data for label: ', training_label
    doc_ids = []
    for clean_excel_file_name in clean_excel_file_names:
        file_path = os.path.join(CLEAN_EXCEL_DATA_DIR, training_label, clean_excel_file_name)
        workbook = open_workbook(file_path)
        worksheet_name = workbook.sheet_names()[0]
        worksheet = workbook.sheet_by_name(worksheet_name)
        current_row = -1
        num_rows = worksheet.nrows - 1
        while current_row < num_rows:
            current_row += 1
            doc_id = worksheet.cell_value(current_row, 0)
            if doc_id:
                doc_ids.append(doc_id)
        print clean_excel_file_name, len(doc_ids)

    results = []
    doc_ids_batch = doc_ids[: BATCH_SIZE]
    chunk_count = 0
    while doc_ids_batch:
        chunk_count += 1
        doc_ids_batch = doc_ids[chunk_count*BATCH_SIZE: (chunk_count+1)*BATCH_SIZE]

        id_query_template = ' OR '.join(['id:%s'] * len(doc_ids_batch))
        id_query = id_query_template % tuple(doc_ids_batch)

        results.extend(s.query(id_query, fields=['id', 'image_url'], rows=len(doc_ids_batch)).results)

    batch_count += 1
    image_sets = [(x['id'], x['image_url']) for x in results]
    count = 0

    for id, image_set in image_sets:

        count += 1
        if not train_category_to_image_map.get(training_label):
            train_category_to_image_map[training_label] = []

        # has all resolutions. we pick the biggest one for best match (hopefully?)
        best_match_image_url = None
        for image in image_set:
            if image.startswith('180x180|'):
                best_match_image_url = image[8:]
                break
        if not best_match_image_url:
            continue
        train_category_to_image_map[training_label].append(best_match_image_url)

    for category, image_urls in train_category_to_image_map.items():
        directory_path = TRAIN_DATA_DIR + '/%s' % category
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        image_label_count = 0
        for image_url in image_urls:
            image_label_count += 1
            if image_label_count >= 2*len(image_urls)/3:
                directory_path = VALIDATION_DATA_DIR + '/%s' % category
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
            try:
                response = requests.get(image_url)
                img = Image.open(StringIO(response.content))
                image_label = category + '_%s.jpg' % image_label_count
                img.save(directory_path + '/%s' % image_label)
            except Exception as ex:
                print ex

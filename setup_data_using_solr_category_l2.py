import solr
from PIL import Image
import requests
from StringIO import StringIO
import os
TRAIN_DATA_DIR = '/Users/saurabhjain/tensorflow/data_clothing_l2_classification/train'
VALIDATE_DATA_DIR = '/Users/saurabhjain/tensorflow/data_clothing_l2_classification/validate'

s = solr.SolrConnection('http://solr-prod.s-9.us:8983/solr/shoprunner')
train_category_to_image_map = {}
MAX_PER_CATEGORY = 200
start = 0
BATCH_SIZE = 1000
batch_count = 0

category_map = {
'Outerwear': ['clothing|Outerwear', 'clothing|Outerwear|blazers', 'clothing|Outerwear|capes & ponchos',
            'clothing|Outerwear|denim jackets', 'clothing|Outerwear|fur & faux fur', 'clothing|Outerwear|leather & faux leather',
            'clothing|Outerwear|quilted & puffers', 'clothing|Outerwear|raincoats', 'clothing|Outerwear|trench coats', 'clothing|Outerwear|vests'],
'activewear': ['clothing|activewear', 'clothing|activewear|active dresses', 'clothing|activewear|active jackets', 'clothing|activewear|active pants',
            'clothing|activewear|active shirts', 'clothing|activewear|active shorts', 'clothing|activewear|active tops',
            'clothing|activewear|active undergarments'],
'coats & jackets': ['clothing|coats & jackets', 'clothing|coats & jackets|blazers', 'clothing|coats & jackets|leather & fur',
                    'clothing|coats & jackets|quilted & puffers', 'clothing|coats & jackets|raincoats', 'clothing|coats & jackets|trench coats',
                    'clothing|coats & jackets|vests'],
'denim': ['clothing|denim', 'denim|bootcut', 'clothing|denim|boyfriend', 'clothing|denim|cropped', 'clothing|denim|flare',
        'clothing|denim|skinny', 'clothing|denim|straight leg'],
'jumpsuits & rompers': ['clothing|jumpsuits & rompers', 'clothing|jumpsuits, rompers & overalls'],
'pants & shorts': ['clothing|pants & shorts', 'clothing|pants & shorts|casual pants', 'clothing|pants & shorts|dress pants',
                    'clothing|pants & shorts|jeans', 'clothing|pants & shorts|leggings', 'clothing|pants & shorts|shorts'],
'pants' : ['clothing|pants', 'clothing|pants|casual pants', 'clothing|pants|denim', 'clothing|pants|denim|bootcut',
            'clothing|pants|denim|boyfriend', 'clothing|pants|denim|cropped', 'clothing|pants|denim|flare', 'clothing|pants|denim|skinny',
            'clothing|pants|denim|straight', 'clothing|pants|dress pants', 'clothing|pants|leggings'],
'shorts': ['clothing|shorts'],
'skirts': ['clothing|skirts', 'clothing|skirts|maxi skirts', 'clothing|skirts|mid length skirts', 'clothing|skirts|mini skirts'],
'dresses': ['clothing|dresses','clothing|dresses|cocktail dresses', 'clothing|dresses|day dresses', 'clothing|dresses|evening dresses', 'clothing|dresses|maxi dresses'],
'sleepwear & loungewear': ['clothing|sleepwear & loungewear', 'clothing|sleepwear & loungewear|loungewear',
                        'clothing|sleepwear & loungewear|loungewear|loungewear bottoms', 'clothing|sleepwear & loungewear|loungewear|loungewear tops',
                        'clothing|sleepwear & loungewear|nightgowns', 'clothing|sleepwear & loungewear|pajamas', 'clothing|sleepwear & loungewear|robes'],
'suits': ['clothing|suits', 'clothing|suits|pant suits', 'clothing|suits|skirt suits', 'clothing|suits|suit separates', 'clothing|suits|tuxedos'],
'sweaters': ['clothing|sweaters', 'clothing|sweaters|cardigans', 'clothing|sweaters|pullovers', 'clothing|sweaters|turtlenecks'],
'swimwear': ['clothing|swimwear', 'clothing|swimwear|cover-ups', 'clothing|swimwear|one-piece swimsuits', 'clothing|swimwear|swim & board shorts',
            'clothing|swimwear|swim bottoms', 'clothing|swimwear|swimsuit tops', 'clothing|swimwear|two-piece swimsuits'],
'tops': ['clothing|tops', 'clothing|tops|blouses', 'clothing|tops|camisoles & tank tops', 'clothing|tops|dress shirts', 'clothing|tops|polo shirts',
        'clothing|tops|sweatshirts', 'clothing|tops|t-shirts'],
'underwear & intimates': ['clothing|underwear & intimates', 'clothing|underwear & intimates|bras', 'clothing|underwear & intimates|bras|demi bras',
                        'clothing|underwear & intimates|bras|full fit bras', 'clothing|underwear & intimates|bras|push up bras',
                        'clothing|underwear & intimates|bras|sports bras', 'clothing|underwear & intimates|bras|strapless & specialty bras',
                        'clothing|underwear & intimates|garters & belts', 'clothing|underwear & intimates|lingerie',
                        'clothing|underwear & intimates|shapewear', 'clothing|underwear & intimates|slips',
                        'clothing|underwear & intimates|undershirts', 'clothing|underwear & intimates|underwear',
                        'clothing|underwear & intimates|underwear|bikinis', 'clothing|underwear & intimates|underwear|boxer briefs',
                        'clothing|underwear & intimates|underwear|boxers', 'clothing|underwear & intimates|underwear|boyshorts',
                        'clothing|underwe3ar & intimates|underwear|briefs', 'clothing|underwear & intimates|underwear|thongs'],
'uniforms': ['clothing|uniforms', 'clothing|uniforms|professional uniforms'],
the kids and accessories are needed in the clothing model because currently, we have accessories and kids stuff categoerized as clothing so tensorflow clothing model needs to know these stuff
'kids & baby': ['kids & baby|girls clothing', 'kids & baby|boys clothing'],
'accessories': ['accessories|hair accessories', 'accessories|headwear & kerchiefs', 'accessories|gloves & mittens', 'accessories|belts & buckles', 'accessories|scarves & shawls', 'accessories|hats', 'accessories|sunglasses', 'accessories|neckties', 'accessories|wallets & money clips', 'accessories|socks & hosiery', 'accessories|watches', 'accessories|handbags', 'accessories|jewelry', 'accessories|umbrellas', 'accessories|luggage & bags', 'accessories|keychains', 'accessories|eyeglasses'],

}


for l1_cat, l2_cats in category_map.items():
    l1_cat = l1_cat.lower()
    print '\n\n\n\n\n Getting images for L1: %s and L2: %s' % (l1_cat, ','.join(l2_cats))
    for l2_cat in l2_cats:
        l2_cat = l2_cat.lower()

        results = s.query(
            'category_autocomplete:"%s"' % l2_cat, fields=['image_url', 'category'], rows=BATCH_SIZE
        ).results
        batch_count += 1
        image_sets = [(x['image_url'], x['category']) for x in results]
        print 'results received from SOLR'
        count = 0
        print 'L2 Cat: ', l2_cat, len(image_sets)
        for image_set, category in image_sets:
            category = category.split('|')[-1]
            # print category
            # if not category or category not in l2_cats:
            #     print 'skipping....'
            #     continue
            count += 1

            if not train_category_to_image_map.get(category):
                train_category_to_image_map[category] = []

            if len(train_category_to_image_map[category]) > MAX_PER_CATEGORY:
                continue

            # has all resolutions. we pick the biggest one for best match (hopefully?)
            best_match_image_url = None
            for image in image_set:
                if image.startswith('180x180|'):
                    best_match_image_url = image[8:]
                    break
            if not best_match_image_url:
                continue
            train_category_to_image_map[category].append(best_match_image_url)
            if count % 100 == 0:
                print count
        for category, image_urls in train_category_to_image_map.items():
            directory_path = TRAIN_DATA_DIR + '/%s' % category
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            image_label_count = 0
            for image_url in image_urls:
                if image_label_count >= (MAX_PER_CATEGORY / 2):
                    directory_path = VALIDATE_DATA_DIR + '/%s' % category
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                image_label_count += 1
                try:
                    response = requests.get(image_url)
                    img = Image.open(StringIO(response.content))
                    image_label = category + '_%s.jpg' % image_label_count
                    img.save(directory_path + '/%s' % image_label)
                except Exception as ex:
                    print ex

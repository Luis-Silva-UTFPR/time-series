import time
import ee
import numpy as np
import datetime
import io
import geopandas as gpd
from shapely.geometry import mapping


def initialize():
    service_account = 'ee-luissilva2024-69a980d6b311.json'
    email = "service-account@ee-luissilva2024.iam.gserviceaccount.com"

    # Autenticação usando o JSON do Service Account
    credentials = ee.ServiceAccountCredentials(email, service_account)
    ee.Initialize(credentials)

initialize()

# Crie o polígono usando ee.Geometry.Polygon
proj = ee.Projection('EPSG:4326').atScale(10).getInfo()

# Get scales out of the transform.
scale_x = proj['transform'][0]
scale_y = -proj['transform'][4]


def extract_ndvi(np_array):
    red_band = np_array['B4']/255  # Banda vermelha (B4)
    nir_band = np_array['B8']/255  # Banda NIR (B8)

    red_band = red_band.astype(float)
    nir_band = nir_band.astype(float)
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    return ndvi


def extract_ndwi(np_array):
    swir1_band = np_array['B11']/255  # Banda swir (B11)
    nir_band = np_array['B8']/255  # Banda NIR (B8)

    swir1_band = swir1_band.astype(float)
    nir_band = nir_band.astype(float)
    ndwi = (nir_band - swir1_band) / (nir_band + swir1_band)
    return ndwi


# Função para extrair os pixels NDVI
def extract_vegetative_bands(image, num_rows, num_cols, region, max_retries=5):
    translate_x = region['coordinates'][0][0][0]
    translate_y = region['coordinates'][0][0][1]
    request = {
        'assetId': image.getInfo()["id"],
        'fileFormat': 'NPY',
        'bandIds': ['B4', 'B8', 'B11'],
        'grid': {
            'dimensions': {
                'width': num_rows,
                'height': num_cols
            },
            'affineTransform': {
                'scaleX': scale_x,
                'shearX': 0,
                'translateX': translate_x,
                'shearY': 0,
                'scaleY': scale_y,
                'translateY': translate_y
            },
            'crsCode': proj['crs'],
        }
    }

    for attempt in range(max_retries):
        try:
            response = ee.data.getPixels(request)
            image_array = np.load(io.BytesIO(response))
            return image_array
        except ee.ee_exception.EEException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise


# Função para extrair os pixels NDVI
def extract_other_bands(image, num_rows, num_cols, region, max_retries=5):
    translate_x = region['coordinates'][0][0][0]
    translate_y = region['coordinates'][0][0][1]
    request = {
        'assetId': image.getInfo()["id"],
        'fileFormat': 'NPY',
        'bandIds': ['B2', 'B3', 'B5', 'B6', 'B7'],
        'grid': {
            'dimensions': {
                'width': num_rows,
                'height': num_cols
            },
            'affineTransform': {
                'scaleX': scale_x,
                'shearX': 0,
                'translateX': translate_x,
                'shearY': 0,
                'scaleY': scale_y,
                'translateY': translate_y
            },
            'crsCode': proj['crs'],
        }
    }

    for attempt in range(max_retries):
        try:
            response = ee.data.getPixels(request)
            image_array = np.load(io.BytesIO(response))
            return image_array
        except ee.ee_exception.EEException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise


# Função para processar cada intervalo de datas
def process_date_range(date_range, region_ee, region):
    ndvi_list = []
    ndwi_list = []
    date_list = []
    other_bands_list = []
    dimensions_list = []  # Lista para armazenar as dimensões de cada imagem

    # Coleção de imagens Sentinel-2
    collection = ee.ImageCollection(
        'COPERNICUS/S2_HARMONIZED'
    ).filterDate(
        date_range[0], date_range[1]
    ).filter(
        ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 0.0001)
    ).filterBounds(
        region_ee
    )

    print(collection.size().getInfo())

    # Iterar sobre a coleção de imagens
    for image in collection.getInfo()['features']:
        image_obj = ee.Image(image['id'])
        num_rows = image_obj.getInfo()['bands'][0]['dimensions'][0]
        num_cols = image_obj.getInfo()['bands'][0]['dimensions'][1]
        dimensions_list.append((num_rows, num_cols))

        vegetative_bands = extract_vegetative_bands(
            image_obj,
            num_rows,
            num_cols,
            region
        )
        print("ok")
        other_bands = extract_other_bands(
            image_obj,
            num_rows,
            num_cols,
            region
        )

        ndvi_list.append(extract_ndvi(vegetative_bands))
        ndwi_list.append(extract_ndwi(vegetative_bands))
        other_bands_list.append(other_bands)
        other_bands_list = np.stack(other_bands_list, axis=0)

        date_list.append(datetime.datetime.utcfromtimestamp(
            image['properties']['system:time_start'] / 1000).strftime(
                '%Y-%m-%d'
            )
        )

    print(other_bands_list.shape)
    return ndvi_list, ndwi_list, other_bands_list, date_list


def iterate_over_cycle(year_1='2017', year_2='2018'):

    gdf = gpd.read_file("dataset_final.gpkg")

    for idx, row in gdf.iterrows():
        region = row.geometry
        region = mapping(region)
        region_ee = ee.Geometry.Polygon(region['coordinates'])
        region_ee = ee.Geometry(region_ee, None, False)

        all_ndvi = []
        all_ndwi = []
        other_bands = []
        all_dates = []

        date_ranges = [
            # Outubro
            (f'{year_1}-10-01', f'{year_1}-10-02'),
            (f'{year_1}-10-03', f'{year_1}-10-04'),
            (f'{year_1}-10-05', f'{year_1}-10-06'),
            (f'{year_1}-10-07', f'{year_1}-10-08'),
            (f'{year_1}-10-09', f'{year_1}-10-10'),
            (f'{year_1}-10-11', f'{year_1}-10-12'),
            (f'{year_1}-10-13', f'{year_1}-10-14'),
            (f'{year_1}-10-15', f'{year_1}-10-16'),

            # Novembro
            # (f'{year_1}-11-01', f'{year_1}-11-07'),
            # (f'{year_1}-11-08', f'{year_1}-11-15'),
            # (f'{year_1}-11-16', f'{year_1}-11-23'),
            # (f'{year_1}-11-24', f'{year_1}-11-30'),

            # # Dezembro
            # (f'{year_1}-12-01', f'{year_1}-12-07'),
            # (f'{year_1}-12-08', f'{year_1}-12-15'),
            # (f'{year_1}-12-16', f'{year_1}-12-23'),
            # (f'{year_1}-12-24', f'{year_1}-12-31'),

            # # Janeiro
            # (f'{year_2}-01-01', f'{year_2}-01-07'),
            # (f'{year_2}-01-08', f'{year_2}-01-15'),
            # (f'{year_2}-01-16', f'{year_2}-01-23'),
            # (f'{year_2}-01-24', f'{year_2}-01-31'),

            # # Fevereiro
            # (f'{year_2}-02-01', f'{year_2}-02-07'),
            # (f'{year_2}-02-08', f'{year_2}-02-15'),
            # (f'{year_2}-02-16', f'{year_2}-02-23'),
            # (f'{year_2}-02-24', f'{year_2}-02-28'),

            # # Março
            # (f'{year_2}-03-01', f'{year_2}-03-07'),
            # (f'{year_2}-03-08', f'{year_2}-03-15'),
            # (f'{year_2}-03-16', f'{year_2}-03-23'),
            # (f'{year_2}-03-24', f'{year_2}-03-31'),

            # # Abril
            # (f'{year_2}-04-01', f'{year_2}-04-07'),
            # (f'{year_2}-04-08', f'{year_2}-04-15'),
            # (f'{year_2}-04-16', f'{year_2}-04-23'),
            # (f'{year_2}-04-24', f'{year_2}-04-30'),
        ]

        # Processar cada intervalo de datas
        for date_range in date_ranges:
            ndvi, ndwi, bands, dates = process_date_range(
                date_range,
                region_ee,
                region
            )
            all_ndvi.extend(ndvi)
            all_ndwi.extend(ndwi)
            other_bands.extend(bands)
            all_dates.extend(dates)

        # Empilhar os NDVI ao longo do eixo 0 para formar o array 4D
        ndvi_array_4d = np.stack(all_ndvi, axis=0)
        # Adicionar uma dimensão adicional no final com tamanho 1
        ndvi_array_4d = np.expand_dims(ndvi_array_4d, axis=-1)

        ndwi_array_4d = np.stack(all_ndwi, axis=0)
        # Adicionar uma dimensão adicional no final com tamanho 1
        ndwi_array_4d = np.expand_dims(ndwi_array_4d, axis=-1)

        print(
            "NDVI Array Shape (serie temporal, eixo y, eixo x, 1):",
            ndvi_array_4d.shape
        )
        np.save(
            f'../workflow/data/bbox/{year_1}_{year_2}/{row["folder_path"]}/ndvi.npy',
            ndvi_array_4d
        )
        np.save(
            f'../workflow/data/bbox/{year_1}_{year_2}/{row["folder_path"]}/ndwi.npy',
            ndwi_array_4d
        )

iterate_over_cycle()

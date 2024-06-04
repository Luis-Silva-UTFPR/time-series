import argparse
import json
import os
import ee
import requests
import io
from datetime import datetime, timedelta
import numpy as np
import geopandas as gpd
from tqdm import tqdm


def initialize():
    service_account = 'ee-luissilva2024-69a980d6b311.json'
    email = "service-account@ee-luissilva2024.iam.gserviceaccount.com"

    # Autenticação usando o JSON do Service Account
    credentials = ee.ServiceAccountCredentials(email, service_account)
    ee.Initialize(credentials)


def calculateNDWI(image):
    return image.normalizedDifference(['B3', 'B8']).rename('NDWI')


def calculateNDVI(image):
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')


def return_date_from_str(image_name: str):
    date_str = image_name.split('_')[0][:8]
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    formatted_date_str = date_obj.strftime("%Y-%m-%d")
    return formatted_date_str


# Função para processar cada intervalo de datas
def process_date_range(
    start,
    end,
    region_ee,
    collection,
    bands_of_interest=['B2', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
):
    ndwi_list = []
    ndvi_list = []
    bands_list = []
    successful_dates = []

    images = collection.toList(collection.size())
    collection = collection.filterDate(
        start,
        end
    )

    size = collection.size().getInfo()

    if size > 0:

        images = collection.toList(collection.size())
        last_image = ee.Image(images.get(-1))
        image_name = last_image.get('system:index').getInfo()
        date_str = return_date_from_str(image_name)

        ndwi = collection.map(calculateNDWI)
        ndwi = ndwi.mean()
        ndvi = collection.map(calculateNDVI)
        ndvi = ndvi.mean()
        image_with_bands = collection.select(bands_of_interest).mean()

        ndwi_id = ee.data.getDownloadId({
            'image': ndwi,
            'bands': ['NDWI'],
            'region': region_ee,
            'scale': 10,
            'format': 'NPY'
        })
        ndvi_id = ee.data.getDownloadId({
            'image': ndvi,
            'bands': ['NDVI'],
            'region': region_ee,
            'scale': 10,
            'format': 'NPY'
        })
        bands_id = ee.data.getDownloadId({
            'image': image_with_bands,
            'bands': ['B2', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'],
            'region': region_ee,
            'scale': 10,
            'format': 'NPY'
        })

        response_ndwi = requests.get(ee.data.makeDownloadUrl(ndwi_id))
        data_ndwi = np.load(io.BytesIO(response_ndwi.content))
        response_ndvi = requests.get(ee.data.makeDownloadUrl(ndvi_id))
        data_ndvi = np.load(io.BytesIO(response_ndvi.content))
        response_bands = requests.get(ee.data.makeDownloadUrl(bands_id))
        data_bands = np.load(io.BytesIO(response_bands.content))

        if data_ndwi.dtype.names:
            regular_ndwi = np.stack(
                [data_ndwi[name] for name in data_ndwi.dtype.names],
                axis=-1
            ).squeeze()
            regular_ndvi = np.stack(
                [data_ndvi[name] for name in data_ndvi.dtype.names],
                axis=-1
            ).squeeze()
            regular_bands = np.stack(
                [data_bands[name] for name in data_bands.dtype.names],
                axis=-1
            ).squeeze()
        else:
            regular_ndwi = np.squeeze(data_ndwi)
            regular_ndvi = np.squeeze(data_ndvi)
            regular_bands = np.squeeze(data_bands)

        ndwi_list.append(regular_ndwi)
        ndvi_list.append(regular_ndvi)
        bands_list.append(regular_bands)
        successful_dates.append(date_str)

    return ndwi_list, ndvi_list, bands_list, successful_dates[0] \
        if len(successful_dates) > 0 else successful_dates


def get_max_ndvi_date(collection, roi):
    def iterate(image, max_dict):
        max_dict = ee.Dictionary(max_dict)
        max_ndvi = ee.Number(max_dict.get('max_ndvi'))
        max_image = ee.Image(max_dict.get('max_image'))

        ndvi = image.select('NDVI').reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=roi,
            scale=10
        ).get('NDVI')

        # Verificar se NDVI é válido
        ndvi = ee.Number(ee.Algorithms.If(ndvi, ndvi, -1))

        # Assegurar que ndvi seja um número antes da comparação
        valid_ndvi = ndvi.gt(-1)

        new_max_image = ee.Image(ee.Algorithms.If(
            valid_ndvi.And(ndvi.gt(max_ndvi)),
            image,
            max_image
        ))

        new_max_ndvi = ee.Number(ee.Algorithms.If(
            valid_ndvi.And(ndvi.gt(max_ndvi)),
            ndvi,
            max_ndvi
        ))

        return ee.Dictionary(
            {
                'max_ndvi': new_max_ndvi,
                'max_image': new_max_image
            }
        )

    initial = ee.Dictionary({'max_ndvi': -1, 'max_image': ee.Image()})
    result = ee.Dictionary(collection.iterate(iterate, initial))
    max_image = ee.Image(result.get('max_image'))

    return max_image


def iterate_over_cycle(start_year, end_year):

    gdf = gpd.read_file("dataset_final.gpkg")
    gdf.to_crs(epsg=4326, inplace=True)

    root_path = f"../data/bbox/{start_year}_{end_year}"
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc='Processing'):

        row_path = row["folder_path"]
        output_path = os.path.join(root_path, row_path)
        data_path = os.path.join(output_path, 'data')

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        xmin, ymin, xmax, ymax = row.geometry.bounds
        roi = ee.Geometry.BBox(xmin, ymin, xmax, ymax)

        all_ndvi = []
        all_ndwi = []
        all_bands = []
        timestamp = []

        initial_date = datetime(int(start_year), 9, 1)
        end_date = datetime(int(end_year), 5, 31)

        end_str = end_date.strftime('%Y-%m-%d')
        initial_str = initial_date.strftime('%Y-%m-%d')

        bands_of_interest = ['B2', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']

        collection = ee.ImageCollection(
            'COPERNICUS/S2_HARMONIZED'
        ).filterBounds(
            roi
        ).filterDate(
            initial_str, end_str
        ).filter(ee.Filter.lt('CLOUD_COVERAGE_ASSESSMENT', 100))

        ndvi_collection = collection.map(calculateNDVI)
        max_ndvi_image = get_max_ndvi_date(ndvi_collection, roi)

        try:
            image_name = max_ndvi_image.get('system:index').getInfo()
            peak_ndvi_date = return_date_from_str(image_name)
        except ee.EEException as e:
            print('Erro ao obter a data da imagem com o NDVI máximo:', e)

        # Processar cada intervalo de datas
        t0 = initial_date
        while t0 <= end_date:
            start, end = t0, t0 + timedelta(days=1)
            ndwi, ndvi, bands, dates = process_date_range(
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d'),
                roi,
                collection,
                bands_of_interest
            )
            all_ndwi.extend(ndwi)
            all_ndvi.extend(ndvi)
            all_bands.extend(bands)
            if len(dates) > 0:
                timestamp.append(dates)
            t0 += timedelta(days=2)

        ndvi_array_4d = np.stack(all_ndvi, axis=0)
        ndvi_array_4d = np.expand_dims(ndvi_array_4d, axis=-1)
        ndwi_array_4d = np.stack(all_ndwi, axis=0)
        ndwi_array_4d = np.expand_dims(ndwi_array_4d, axis=-1)
        bands_array_4d = np.stack(all_bands, axis=0)

        timestamp_dict = {
            "timestamp": timestamp,
            "peak_ndvi": peak_ndvi_date
        }

        np.save(os.path.join(data_path, 'NDVI.npy'), ndvi_array_4d)
        np.save(os.path.join(data_path, 'NDWI.npy'), ndwi_array_4d)
        np.save(os.path.join(data_path, 'BANDS.npy'), bands_array_4d)

        with open(f"{output_path}/bands_of_interest.json", "w") as json_file:
            json.dump(bands_of_interest, json_file)
        with open(f"{output_path}/timestamp.json", "w") as json_file:
            json.dump(timestamp_dict, json_file)


def main():

    parser = argparse.ArgumentParser(
        description="baixar imagens com base no ano safra"
    )
    parser.add_argument("start_year", type=int, help="start year")
    parser.add_argument("end_year", type=int, help="end year")

    args = parser.parse_args()

    initialize()
    iterate_over_cycle(args.start_year, args.end_year)


if __name__ == "__main__":
    main()

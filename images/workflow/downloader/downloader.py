import argparse
import json
import os
import ee
import io
import numpy as np
import geopandas as gpd
from tqdm import tqdm


def initialize():
    service_account = 'ee-luissilva2024-69a980d6b311.json'
    email = "service-account@ee-luissilva2024.iam.gserviceaccount.com"

    # Autenticação usando o JSON do Service Account
    credentials = ee.ServiceAccountCredentials(email, service_account)
    ee.Initialize(credentials)


def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([ndvi, ndwi])


def get_bert_from_image(image):
    image = ee.Image(image)
    doy = image.date().getRelative("day", "year")
    return doy


def ee_largest_geometry_centroid(geometry):
    def largest_geometry(geometry):
        geoms = geometry.geometries()
        areas = geoms.map(lambda geom: ee.Geometry(geom).area())
        return ee.Geometry(geoms.get(
            areas.indexOf(areas.sort().reverse().get(0))
        ))

    centroid = ee.Geometry(
        ee.Algorithms.If(
            geometry.type().equals("MultiPolygon"),
            largest_geometry(geometry).centroid(),
            geometry.centroid(),
        )
    )

    return centroid


def get_max_ndvi_date(collection, roi):
    def iterate(image, max_dict):
        max_dict = ee.Dictionary(max_dict)
        max_ndvi = ee.Number(max_dict.get('max_ndvi'))
        max_image = ee.Image(max_dict.get('max_image'))
        max_doy = ee.Number(max_dict.get('max_doy'))

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

        new_max_doy = ee.Number(ee.Algorithms.If(
            valid_ndvi.And(ndvi.gt(max_ndvi)),
            image.date().getRelative('day', 'year'),
            max_doy
        ))

        return ee.Dictionary(
            {
                'max_ndvi': new_max_ndvi,
                'max_image': new_max_image,
                'max_doy': new_max_doy
            }
        )

    initial = ee.Dictionary(
        {'max_ndvi': -1, 'max_image': ee.Image(), 'max_doy': -1}
    )
    result = ee.Dictionary(collection.iterate(iterate, initial))
    max_doy = ee.Number(result.get('max_doy')).getInfo()

    return max_doy


def download_former_images(
    geometry, start_date, end_date, crop_type, download_filepath
):
    CLOUD_PERCENTAGE_TOP_LIMIT = 100
    CHIP_SIZE = 5
    CROP_REMAP_DICT = {
        "SOYBEAN": 2,
        "CORN": 1,
        "WHEAT": 3,
    }

    geometry = ee.Geometry(geometry.__geo_interface__)
    start_date = ee.Date(start_date)
    end_date = ee.Date(end_date)

    geometry = (
        ee_largest_geometry_centroid(geometry).buffer(
            CHIP_SIZE * 5 + -2
        ).bounds()
    )
    geometry_mask = ee.Image(1).clip(geometry)

    filter = ee.Filter.And(
        ee.Filter.date(start_date, end_date),
        ee.Filter.bounds(geometry),
    )
    boa_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filter(filter)
        .map(add_indices)
        .select(
            [
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B11",
                "B12",
                "NDVI",
                "NDWI"
            ]
        )
    )
    max_doy = get_max_ndvi_date(boa_collection, geometry)
    print(max_doy)

    cloud_mask = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filter(filter)
        .select(["probability"], ["cloud"])
    )

    boa_collection = boa_collection.combine(cloud_mask)
    boa_collection = boa_collection.map(
        lambda image: image.set("time_dummy", image.date().format("YYYYMMdd"))
    ).distinct("time_dummy")  # Keeping only one image per day

    def get_cloud_percentage(image):
        image = ee.Image(image).updateMask(geometry_mask)

        cloud_percentage = ee.Number(
            image.select(["cloud"])
            .unmask(0, False)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10,
                maxPixels=1e13
            )
            .get("cloud")
        )

        return image.set("cloud_percentage", cloud_percentage).select(
            image.bandNames().remove("cloud")
        )

    boa_collection = boa_collection.map(get_cloud_percentage).filter(
        ee.Filter.lt("cloud_percentage", CLOUD_PERCENTAGE_TOP_LIMIT)
    )

    image_indexes = ee.data.computeValue(
        boa_collection.aggregate_array("system:index")
    )

    if len(image_indexes) < 5:
        return

    ts = np.empty([len(image_indexes), 12, CHIP_SIZE, CHIP_SIZE])
    doy = np.array(
        ee.data.computeValue(
            boa_collection.toList(10000).map(get_bert_from_image)
        )
    ).astype(np.int16)

    proj = ee.data.computeValue(ee.Projection("EPSG:4326").atScale(10))
    scale_y = -proj["transform"][0]
    scale_x = proj["transform"][4]

    listCoords = ee.data.computeValue(ee.Array.cat(geometry.coordinates(), 1))

    xMin = listCoords[0][0]
    yMax = listCoords[2][1]
    coords = [xMin, yMax]

    for n, image_index in enumerate(image_indexes):
        image = ee.data.computePixels(
            {
                "expression": ee.Image(
                    boa_collection.filter(
                        ee.Filter.eq("system:index", image_index)
                    ).first()
                ),
                "fileFormat": "NPY",
                "grid": {
                    "dimensions": {"width": CHIP_SIZE, "height": CHIP_SIZE},
                    "affineTransform": {
                        "scaleX": scale_x,
                        "scaleY": scale_y,
                        "translateX": coords[0],
                        "translateY": coords[1],
                    },
                    "crsCode": "EPSG:4326",
                },
            }
        )
        image = np.load(io.BytesIO(image)).tolist()
        image = np.moveaxis(image, 2, 0)  # Changing to channels-first

        ts[n, :, :, :] = image

    filtered = (
        (np.mean(
            ts[:, 0, :, :], axis=(1, 2)
        ) < 5000)  # Corte seco de nuvem
        & (np.mean(
            ts[:, 6, :, :], axis=(1, 2)
        ) > 1500)  # Corte seco de sombra de nuvem
        & (np.all(
            ts, axis=(1, 2, 3)
        ))  # Removendo inconsistências do Sentinel 2
    )
    ts = ts[filtered]
    doy = doy[filtered]
    timestamp_dict = {
        "timestamp": doy.tolist()
    }

    if ts.shape[0] >= 5:
        np.savez_compressed(
            os.path.join(download_filepath, f"{crop_type}.npz"),
            ts=ts.astype(np.int16),
            doy=doy.astype(np.int16),
            class_label=np.array(CROP_REMAP_DICT[crop_type]).astype(np.int16),
        )
        with open(f"{download_filepath}/timestamp.json", "w") as json_file:
            json.dump(timestamp_dict, json_file)


def batch_images_download(df):  # Known bug: crashes python when interrupted

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Processing'):
        if not os.path.exists(row.downloaded_filepath):
            os.makedirs(row.downloaded_filepath)
        download_former_images(
            row.geometry,
            row.start_date,
            row.end_date,
            row.crop_type,
            row.downloaded_filepath
        )


def main():

    parser = argparse.ArgumentParser(
        description="baixar imagens com base no ano safra"
    )
    parser.add_argument("gdf", type=str, help="geodataframe path")

    args = parser.parse_args()
    gdf = gpd.read_parquet(f"{args.gdf}.parquet")
    gdf["downloaded_filepath"] = gdf.apply(
        lambda row: f"data/images/{'_'.join(map(str, [row.crop_type, row.id]))}",
        axis=1,
    )

    initialize()
    batch_images_download(gdf)
    gdf.to_parquet(f"{args.gdf}_downloaded.parquet")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

import pprint
import fiona
import logging
import time
import csv
import sys
import os
from shapely.geometry import Point, shape
from collections import Counter

CLOSE_DIST_THRESHOLD = 0.1  # deg
# logging.basicConfig(level=logging.INFO)


class Mosaic:
    def __init__(self,
                 shapefile_path='../qa_tests_data/mosaic/ModelBoundaries.shp'):
        self.shapefile_path = shapefile_path

    def get_models_list(self):
        """
        Returns a list of all models in the shapefile
        """
        with fiona.open(self.shapefile_path, 'r') as shp:
            models = [polygon['properties']['code'] for polygon in shp]
        return models

    def get_model_by_lon_lat(self, lon, lat):
        """
        Given a longitude and latitude, finds the corresponding hazard model
        in the global mosaic.

        :param lon:
            The site longitude
        :param lat:
            The site latitude
        """
        t0 = time.time()
        lon = float(lon)
        lat = float(lat)
        point = Point(lon, lat)

        # # If we prefer to make a point-in-polygon search:
        # models = []
        with fiona.open(self.shapefile_path, 'r') as shp:
            # # If we prefer to make a point-in-polygon search:
            # models.extend([polygon['properties']['code']
            #                for polygon in shp
            #                if point.within(shape(polygon['geometry']))])

            # NOTE: poly.distance(point) returns 0.0 if point is within poly
            #       To calculate the distance to the nearest edge, one would do
            #       poly.exterior.distance(point) instead
            model_dist = {
                polygon['properties']['code']:
                    shape(polygon['geometry']).distance(point)
                for polygon in shp
            }
        close_models = {
            model: model_dist[model]
            for model in model_dist
            if model_dist[model] < CLOSE_DIST_THRESHOLD
        }
        num_close_models = len(close_models)
        if num_close_models < 1:
            logging.error(
                f'Site at lon={lon} lat={lat} is not covered by any model!')
            return
        elif num_close_models > 1:
            model = min(close_models, key=close_models.get)
            logging.warning(
                f'Site at lon={lon} lat={lat} is on the border between more'
                f' than one model: {close_models}. Using {model}')
        else:  # only one close model was found
            model = list(close_models)[0]
            logging.info(
                f'Site at lon={lon} lat={lat} is covered by model {model}'
                f' (distance: {model_dist[model]})')
        logging.debug(f'Model search took {time.time() - t0} seconds')
        return model

    def get_models_by_sites_csv(self, csv_path):
        """
        Given a csv file with (Longitude, Latitude) of sites, returns a
        dictionary having as key the site location and as value the mosaic
        model that covers that site

        :param csv_path:
            path of the csv file containing sites coordinates
        """
        model_by_site = {}
        with open(csv_path, 'r') as sites:
            for site in csv.DictReader(sites):
                try:
                    lon = site['Longitude']
                    lat = site['Latitude']
                except KeyError:
                    lon = site['lon']
                    lat = site['lat']
                model_by_site[(lon, lat)] = self.get_model_by_lon_lat(lon, lat)
        logging.info(Counter(model_by_site.values()))
        return model_by_site


if __name__ == '__main__':
    try:
        csv_path = sys.argv[1]
    except IndexError:
        print('Please provide the path of a csv file with site coordinates')
        exit(1)
    if not os.path.isfile(csv_path):
        print(f'The path {csv_path} does not correpond to a valid file')
        exit(1)
    pprint.pprint(Mosaic().get_models_by_sites_csv(csv_path))

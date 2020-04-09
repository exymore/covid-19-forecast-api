import pandas as pd
import json
from rest_framework import viewsets, mixins, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import CountryNameSerializer


def generate_file_path(country):
    return f'covid_19_app/forecast/data/covid19_forecast_data_{country}_cases.csv'


@api_view(['GET'])
def stats_per_country(request):
    serializer = CountryNameSerializer(data=request.query_params)
    serializer.is_valid(raise_exception=400)
    country = serializer.validated_data['country']
    filePath = generate_file_path(country)
    try:
        df = pd.read_csv(filePath, parse_dates=True)
        df = df.drop(df.columns[[0]], axis=1)
        result = json.loads(df.to_json(orient='records', date_format='iso'))
        return Response(result, status.HTTP_200_OK)
    except FileNotFoundError:
        return Response(status=status.HTTP_400_BAD_REQUEST)
    return Response(status=status.HTTP_400_BAD_REQUEST)

from rest_framework import serializers


class CountryNameSerializer(serializers.Serializer):
    country = serializers.CharField(max_length=255)

from .views import stats_per_country
from django.urls import path, include


urlpatterns = [
    path('forecast', stats_per_country, name='stats'),  # get
]

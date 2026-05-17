from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserProfileViewSet
from .temporal_views import TemporalViewSet

router = DefaultRouter()
router.register(r'profile', UserProfileViewSet, basename='profile')
router.register(r'temporal', TemporalViewSet, basename='temporal')

urlpatterns = [
    path('', include(router.urls)),
]

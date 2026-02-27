"""
ServVia 3.0 API URL Configuration
==================================

Routes all chat traffic through the new ServVia pipeline (api/views.py).
Legacy agricultural endpoints are preserved under a separate namespace.
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from api.views import ServViaChatViewSet

router = DefaultRouter()
router.register(r"chat", ServViaChatViewSet, basename="chat")

urlpatterns = [
    path("", include(router.urls)),
]

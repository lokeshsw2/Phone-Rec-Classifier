from django.urls import path, include
from . import views
urlpatterns = [
    path('', views.all_products, name='all_products'),
    path('product_details', views.product_detail, name='product_detail'),
    path('classifier', views.classify, name='classify'),
    path('opt', views.opt, name= 'opt')
]
from django.test import TestCase
from PhoneRecc.views import return_image_list, return_title_image_list

# Create your tests here.

#python manage.py test PhoneRecc

class PhoneTest(TestCase):

    def test_case_1(self):
        response = self.client.get('')
        self.assertEqual(response.status_code, 200)

    def test_case_2(self):
        response = self.client.get('/classifier')
        self.assertEqual(response.status_code, 200)

    def test_case_3(self):
        response = self.client.get('')
        self.assertTemplateUsed(response, 'products.html')

    def test_image_list_testing(self):
        test_list = ['A', 'B', 'C']
        expected_output = [['A', 0], ['B', 1], ['C', 2]]
        self.assertListEqual(expected_output, return_image_list(test_list))

    def test_title_image_list_testing(self):
        test_list1 = [['A', 0], ['B', 1], ['C', 2]]
        test_list2 = ['X','Y','Z']
        expected_output = [['A','X',0], ['B','Y',1], ['C','Z',2]]
        self.assertListEqual(expected_output, return_title_image_list(test_list1, test_list2))


    def test_title_image_list_corner_testing(self):
        test_list1 = [['A', 0], ['B', 1]]
        test_list2 = ['X','Y','Z']
        expected_output = test_list1
        self.assertListEqual(expected_output, return_title_image_list(test_list1, test_list2))

    def test_null_image_list_testing(self):
        test_list = ['A', 'B', '']
        expected_output = []
        self.assertListEqual(expected_output, return_image_list(test_list))









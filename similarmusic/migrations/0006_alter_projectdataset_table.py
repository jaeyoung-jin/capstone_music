# Generated by Django 4.0.4 on 2022-05-10 07:40

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('similarmusic', '0005_delete_dataset'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='projectdataset',
            table='dataset',
        ),
    ]

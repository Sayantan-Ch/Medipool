# Generated by Django 4.1.2 on 2022-10-14 18:50

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Doctor',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('speciality', models.CharField(max_length=100)),
                ('practice_name', models.CharField(max_length=100)),
                ('address', models.CharField(max_length=100)),
                ('pincode', models.IntegerField()),
            ],
        ),
    ]

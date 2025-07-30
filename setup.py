from setuptools import setup, find_packages

setup(
    name='mosaic_damage_detection',
    version='0.1.0',
    author='Anthony Stan',
    description='Detect and analyze mosaic tile damage using LPIPS and image comparison',
    packages=find_packages(),
    install_requires=[
        'lpips',
        'torch',
        'torchvision',
        'Pillow',
        'opencv-python',
        'matplotlib',
        'numpy',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'detect-damage=lpips_img:main',
            'analyze-damage=analyze_lpips_results:main',
        ]
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
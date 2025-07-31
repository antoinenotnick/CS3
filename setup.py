from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name='mosaic_damage_detection',
    version='0.1.0',
    author='Anthony Stan, Rishith Arra',
    author_email=['anthonytstan01@gmail.com', 'rishitharra27@gmail.com'],
    description='Detect and analyze mosaic tile damage using LPIPS and image comparison with person detection',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/antoinenotnick/CS3',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'flake8>=3.8',
        ],
        'analysis': [
            'jupyter>=1.0.0',
            'seaborn>=0.11.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'detect-damage=lpips_img_test:main',
            'analyze-damage=analysis:main',
            'monitor-camera=lpips_vid:main',
            'test-detection=rfdetr_test:main',
        ]
    },
    include_package_data=True,
    package_data={
        'mosaic_damage_detection': [
            'images/**/*',
            'patches/**/*',
            '*.csv',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    keywords='image-processing computer-vision damage-detection lpips mosaic tiles',
)
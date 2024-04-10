from setuptools import setup

setup(name='lm-flash-bench',
      version='1.0.0',
      author='Krzysztof (Chris) Ociepa',
      packages=['lm-flash-bench'],
      description='Simple, customizable, and rapid benchmarking framework for fine-tuned LLMs',
      license='MIT',
      install_requires=[
            'torch',
            'transformers',
      ],
)

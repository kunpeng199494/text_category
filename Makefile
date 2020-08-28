clean:
	rm -rf build dist *.egg-info

uninstall:
	pip uninstall text_category -y

install:
	python setup.py install

reinstall:
	make clean && make install

build:
	python setup.py sdist bdist_wheel

upload:
	twine upload -r bailian dist/*

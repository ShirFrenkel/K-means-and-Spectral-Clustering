from invoke import task

@task
def build(c):
    c.run("python setup.py build_ext --inplace")

@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")

@task(build)
def run(c, n=0, k=0, Random=True):
    from main import main  # this is here because it must be after build is called
    main(Random, n, k)





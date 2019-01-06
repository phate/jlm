struct struct1 {
	int x;
};

struct struct2 {
	struct struct1 * pdfa;
};

static struct struct1 g1 = {0};

struct struct2 g2 = {&g1};

struct struct1 g3, *p = &g3;

int
main()
{
	return 0;
}

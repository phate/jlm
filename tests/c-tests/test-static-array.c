#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char * names[][10] = { { "a", "b", "c", "d", "e", "f", "g", NULL },
                                    { "x", "y", "z", NULL },
                                    { "h", "i", "j", "k", "l", NULL },
                                    { "m", NULL },
                                    { "n", "o", "p", "q", "r", NULL } };

static void
print_names()
{
  for (int i = 0; i < sizeof(names) / sizeof(names[0]); i++)
  {
    for (int k = 0; names[i][k] != NULL; k++)
    {
      printf("%s\n", names[i][k]);
    }
  }
}

static void
get_a(char ** name, int * length)
{
  *name = (char *)names[0][0];
  *length = sizeof(*name);
}

static void
get_z(char ** name, int * length)
{
  *name = (char *)names[1][2];
  *length = sizeof(*name);
}

static void
get_q(char ** name, int * length)
{
  *name = (char *)names[4][3];
  *length = sizeof(*name);
}

int
main()
{
  print_names();

  {
    char * name;
    int length;
    get_a(&name, &length);
    assert(strcmp(name, "a") == 0);
  }

  {
    char * name;
    int length;
    get_z(&name, &length);
    assert(strcmp(name, "z") == 0);
  }

  {
    char * name;
    int length;
    get_q(&name, &length);
    assert(strcmp(name, "q") == 0);
  }
}

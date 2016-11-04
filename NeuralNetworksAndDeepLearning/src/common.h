/*
 * common.h
 *
 *  Created on: 2016. 8. 27.
 *      Author: jhkim
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <limits.h>

#include <iostream>

#define ALIGNUP(x, n)               ((~(n-1))&((x)+(n-1)))
#define ALIGNDOWN(x, n)             ((~(n-1))&(x))

#endif /* COMMON_H_ */

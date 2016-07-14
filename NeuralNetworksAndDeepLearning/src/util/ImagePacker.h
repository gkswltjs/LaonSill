/*
 * ImagePacker.h
 *
 *  Created on: 2016. 7. 13.
 *      Author: jhkim
 */

#ifndef IMAGEPACKER_H_
#define IMAGEPACKER_H_

class ImagePacker {
public:
	ImagePacker(const char *image_dir, const char *outfile_image_path, const char *outfile_label_path);
	virtual ~ImagePacker();

	void pack();

private:
	char image_dir[256];
	char outfile_image_path[256];
	char outfile_label_path[256];

};

#endif /* IMAGEPACKER_H_ */

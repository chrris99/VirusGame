//=============================================================================================
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Conforti Christian
// Neptun : F8R430
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

const int TESSELATION_LEVEL = 30;
const float PI_F = 3.14159265358979f;

#pragma region DualNumber

template<class T>
struct DualNumber {
	float funcVal;
	T derivative;

	DualNumber(float f0 = 0, T d0 = T(0)) {
		funcVal = f0;
		derivative = d0;
	}

	DualNumber operator+(const DualNumber& rhs) const {
		return DualNumber(funcVal + rhs.funcVal, derivative + rhs.derivative);
	}

	DualNumber operator-(const DualNumber& rhs) const {
		return DualNumber(funcVal - rhs.funcVal, derivative - rhs.derivative);
	}

	DualNumber operator*(const DualNumber& rhs) const {
		return DualNumber(funcVal * rhs.funcVal, funcVal * rhs.derivative + derivative * rhs.funcVal);
	}

	DualNumber operator/(const DualNumber& rhs) const {
		return DualNumber(funcVal / rhs.funcVal, (rhs.funcVal * derivative - rhs.derivative * funcVal) / (rhs.funcVal * rhs.funcVal));
	}
};

template<class T> DualNumber<T> Exp(DualNumber<T> g) { return DualNumber<T>(expf(g.funcVal), expf(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Sin(DualNumber<T> g) { return DualNumber<T>(sinf(g.funcVal), cosf(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Cos(DualNumber<T> g) { return DualNumber<T>(cosf(g.funcVal), -sinf(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Sinh(DualNumber<T> g) { return DualNumber<T>(sinh(g.funcVal), cosh(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Cosh(DualNumber<T> g) { return DualNumber<T>(cosh(g.funcVal), sinh(g.funcVal) * g.derivative); }
template<class T> DualNumber<T> Tanh(DualNumber<T> g) { return Sinh(g) / Cosh(g); }

using DualNumber2 = DualNumber<vec2>;

#pragma endregion

#pragma region Geometry

struct VertexData {
	vec3 position, normal;
	vec2 textureCoordinate;
};

class Geometry {
protected:
	unsigned int vao;	// Vertex Array Object 
	unsigned int vbo;	// Vertex Buffer Object 

public:
	Geometry() {
		glGenVertexArrays(1, &vao); // Can't be global variable
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}

	virtual void draw() = 0;

	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParametricSurface : public Geometry {
private:
	unsigned int vertexPerStrip, nStrips;

public:
	ParametricSurface() {
		vertexPerStrip = 0;
		nStrips = 0;
	}

	virtual void evaluate(DualNumber2& U, DualNumber2& V, DualNumber2& X, DualNumber2& Y, DualNumber2& Z) = 0;

	VertexData genVertexData(const float u, const float v) {
		VertexData vertexData;
		vertexData.textureCoordinate = vec2(u, v);

		DualNumber2 X, Y, Z;
		DualNumber2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		evaluate(U, V, X, Y, Z);

		vertexData.position = vec3(X.funcVal, Y.funcVal, Z.funcVal);
		vertexData.normal = cross(vec3{ X.derivative.x, Y.derivative.x, Z.derivative.x }, vec3{ X.derivative.y, Y.derivative.y, Z.derivative.y });

		return vertexData;
	}
	
	void create(const int N = TESSELATION_LEVEL, const int M = TESSELATION_LEVEL) {
		vertexPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vertices;
		
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vertices.push_back(genVertexData((float)j / M, (float)i / N));
				vertices.push_back(genVertexData((float)j / M, (float)(i + 1) / N));
			}
		}

		glBufferData(GL_ARRAY_BUFFER, vertexPerStrip * nStrips * sizeof(VertexData), &vertices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);	// Attribute Array 0 = Position
		glEnableVertexAttribArray(1);	// Attribute Array 1 = Normal Vector
		glEnableVertexAttribArray(2);	// Attribute Array 2 = Texture Coordinate

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(offsetof(VertexData, position)));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(offsetof(VertexData, normal)));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(offsetof(VertexData, textureCoordinate)));
	}

	void draw() override {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) {
			glDrawArrays(GL_TRIANGLE_STRIP, i * vertexPerStrip, vertexPerStrip);
		}
	}
};

class Sphere final : public ParametricSurface {
public:
	Sphere() { create(); }

	void evaluate(DualNumber2& U, DualNumber2& V, DualNumber2& X, DualNumber2& Y, DualNumber2& Z) override final {
		U = U * 2.0f * (float)M_PI;
		V = V * (float)M_PI;
		X = Cos(U) * Sin(V);
		Y = Sin(U) * Sin(V);
		Z = Cos(V);
	}
};

class Tractricoid final : public ParametricSurface {
public:
	Tractricoid() { create(); }

	void evaluate(DualNumber2& U, DualNumber2& V, DualNumber2& X, DualNumber2& Y, DualNumber2& Z) override final {
		const float height = 3.0f;
		U = U * height;
		V = V * 2.0f * M_PI;
		X = Cos(V) / Cosh(U);
		Y = Sin(V) / Cosh(U);
		Z = U - Tanh(U);
	}
};

#pragma endregion

// Rough Material
struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

#pragma region Camera

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;

public:

	/**
	 *
	 */
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	}

	/**
	 *
	 */
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(vec3{ -wEye.x, -wEye.y, -wEye.z }) * mat4 {
				{ u.x, v.x, w.x, 0 },
				{ u.y, v.y, w.y, 0 },
				{ u.z, v.z, w.z, 0 },
				{ 0, 0, 0, 1 }
		};
	}

	/**
	 *
	 */
	mat4 P() {
		float sy = 1 / tanf(fov / 2);
		return mat4{
			{sy / asp, 0, 0, 0},
			{0, sy, 0, 0},
			{0, 0, -(fp + bp) / (bp - fp), -1},
			{0, 0, -2 * fp * bp / (bp - fp), 0}
		};
	}
};

#pragma endregion



#pragma region Texture

/**
 *
 */
class CheckerBoardTexture : public Texture {
public:
	CheckerBoardTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4& yellow = vec4{ 1, 1, 0, 1 };
		const vec4& blue = vec4{ 0, 0, 1, 1 };
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);

	}
};

class StripeTexture : public Texture {
public:
	StripeTexture(const int width, const int height, const vec4& color1 = vec4{ 1, 1, 0, 1 }, const vec4& color2 = vec4{ 0, 0, 1, 1 }) : Texture() {
		std::vector<vec4> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = y % 2 ? color1 : color2;
		}
		create(width, height, image, GL_NEAREST);
	}
};


#pragma endregion

#pragma region Shaders

/*
Interface az objektumok és a shaderek között
Az objektum azon változói amiket szeretne érvényesíteni 
a shaderben
A shader innen szedi ki a dolgokat
*/
struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3 wEye;
};

class PhongShader : public GPUProgram {
private:
	const char* vertexSource = R"(
		#version 330
		precision highp float;
		
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4 MVP, M, Minv;
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3 wEye;

		layout(location = 0) in vec3 vtxPos;
		layout(location = 1) in vec3 vtxNorm;
		layout(location = 2) in vec2 vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
			wView = wEye * wPos.w - wPos.xyz;
			wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
			texcoord = vtxUV;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int nLights;
		uniform sampler2D diffuseTexture;

		in vec3 wNormal;
		in vec3 wView;
		in vec3 wLight[8];
		in vec2 texcoord;

		out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;
			
			vec3 radiance = vec3(0, 0, 0);	
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
				radiance += ka * lights[i].La +
							(kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";

public:
	PhongShader() {
		create(vertexSource, fragmentSource, "fragmentColor");
	}

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}

	void bind(const RenderState& state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

PhongShader* gpuProgram;

#pragma endregion

struct Object {
	Material* material;
	Texture* texture;
	Geometry* geometry;

	// Model transformation parameters
	vec3 scale, translation, rotationAxis;
	float rotationAngle;

	Object(Material* _material, Texture* _texture, Geometry* _geometry) :
		scale{ vec3{ 1, 1, 1 } }, translation{ vec3{ 0, 0, 0 } }, rotationAxis{ vec3{ 0, 0, 1 } }, rotationAngle{ 0 } {
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	/**
	 * 
	 * @param M
	 * @param Minv
	 */
	virtual void setModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3{ 1 / scale.x, 1 / scale.y, 1 / scale.z });
	}

	/**
	 *
	 * @pram state
	 */
	void draw(RenderState state) {
		mat4 M, Minv;
		setModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		gpuProgram->bind(state);
		geometry->draw();
	}

	/**
	 *
	 * @param tstart
	 * @param tend
	 */
	virtual void animate(const float t) {
		rotationAngle = 0.8f * t;
	}
};

class Virus : public Object {
private:
	Sphere* body;
	std::vector<Tractricoid*> corona;

public:

};

/**
 * Represents the virtual world
 */
class Scene {
private:
	Camera camera;
	std::vector<Object*> objects;
	std::vector<Light> lights;

public:

	/**
	 *
	 */
	void build() {

		// Materials
		Material* material0 = new Material();
		material0->kd = vec3{ 0.6f, 0.4f, 0.2f };
		material0->ks = vec3{ 4, 4, 4 };
		material0->ka = vec3{ 0.1f, 0.1f, 0.1f };
		material0->shininess = 100;

		Material* material1 = new Material();
		material1->kd = vec3{ 0.8f, 0.6f, 0.4f };
		material1->ks = vec3{ 0.3f, 0.3f, 0.3f };
		material1->ka = vec3{ 0.2f, 0.2f, 0.2f };
		material1->shininess = 30;

		// Textures
		Texture* texture1 = new StripeTexture(4, 8);
		Texture* texture2 = new StripeTexture(15, 20);

		// Geometries
		Geometry* sphere = new Sphere();
		Geometry* tractricoid = new Tractricoid();

		Object* sphereObject = new Object(material0, texture2, sphere);
		sphereObject->translation = vec3{ 3, 1, 0 };
		sphereObject->scale = vec3{ 1.0f, 1.0f, 1.0f };
		sphereObject->rotationAxis = vec3{ 1, 0, 0 };
		objects.push_back(sphereObject);

		Object* tractiObject = new Object(material0, texture1, tractricoid);
		tractiObject->translation = vec3{ -4, 3, 0 };
		tractiObject->rotationAxis = vec3{ 1, 0, 0 };
		objects.push_back(tractiObject);

		// Camera
		camera.wEye = vec3{ 0, 0, 8 };
		camera.wLookat = vec3{ 0, 0, 0 };
		camera.wVup = vec3{ 0, 1, 0 };

		// Lights
		lights.resize(3);
		lights[0].wLightPos = vec4{ 5, 5, 4, 0 };
		lights[0].La = vec3{ 0.1f, 0.1f, 1 };
		lights[0].Le = vec3{ 3, 0, 0 };

		lights[1].wLightPos = vec4{ 5, 10, 20, 0 };
		lights[1].La = vec3{ 0.2f, 0.2f, 0.2f };
		lights[1].Le = vec3{ 0, 3, 0 };

		lights[2].wLightPos = vec4{ -5, 5, 5, 0 };
		lights[2].La = vec3{ 0.1f, 0.1f, 0.1f };
		lights[2].Le = vec3{ 0, 0, 3 };
	}

	void render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (auto object : objects) object->draw(state);
	}

	void animate(const float t) {
		for (auto object : objects) object->animate(t);
	}
};

Scene scene;

# pragma region EventHandling

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	gpuProgram = new PhongShader();
	scene.build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);				// clear frame buffer

	scene.render();

	glutSwapBuffers(); // exchange buffers for double buffering
}

void onKeyboard(unsigned char key, int pX, int pY) { }

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouseMotion(int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onIdle() {
	// Discrete time simulation
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;

	// Conversion to seconds
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.animate(t);
	}

	// Redrawing the display
	glutPostRedisplay();
}

#pragma endregion
